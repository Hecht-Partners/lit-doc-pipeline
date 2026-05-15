"""
LLM enrichment for litigation document chunks.

Adds summaries, key quotes, categorization, relevance scoring, and claims
tracking to existing chunks. Supports three backends:
  - Ollama (local, no API key needed)
  - Anthropic (Claude API, requires API key)
  - Claude Code (interactive, via background Task agents)

CRITICAL: All key quotes are validated as exact substrings of core_text.
Hallucinated quotes are discarded per TRD Section 9.4.
"""

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from tqdm import tqdm

from lit_pipeline.citation_types import Chunk, DocumentType

logger = logging.getLogger(__name__)


def _should_disable_tqdm():
    """Check if progress bars should be disabled."""
    return (os.environ.get('TQDM_DISABLE', '0') == '1' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            os.environ.get('CI', '').lower() == 'true')


# Valid categories for chunk classification
VALID_CATEGORIES = {
    "legal_argument",
    "factual_allegation",
    "evidence_reference",
    "expert_opinion",
    "witness_statement",
    "damages_analysis",
    "procedural",
    "background",
    "statutory_regulatory",
    "case_citation",
    "claim_construction",
    "infringement_analysis",
    "validity_invalidity",
    "unclassified",
}

VALID_RELEVANCE = {"high", "medium", "low"}

# Default models per backend
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class CaseContext:
    """Context about the litigation case for enrichment prompts."""
    case_type: str = "patent"
    parties: List[str] = field(default_factory=list)
    patents: List[str] = field(default_factory=list)
    claims_at_issue: List[int] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        parts = [f"Case type: {self.case_type}"]
        if self.parties:
            parts.append(f"Parties: {', '.join(self.parties)}")
        if self.patents:
            parts.append(f"Patents: {', '.join(self.patents)}")
        if self.claims_at_issue:
            parts.append(f"Claims at issue: {', '.join(map(str, self.claims_at_issue))}")
        return "\n".join(parts)


@dataclass
class EnrichmentResult:
    """Per-chunk enrichment output."""
    chunk_id: str
    summary: Optional[str] = None
    key_quotes: Optional[List[str]] = None
    category: str = "unclassified"
    relevance_score: str = "medium"
    claims_addressed: Optional[List[int]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class EnrichmentStats:
    """Run-level enrichment statistics."""
    total: int = 0
    enriched: int = 0
    skipped: int = 0
    failed: int = 0
    quotes_validated: int = 0
    quotes_discarded: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Enrichment: {self.enriched}/{self.total} enriched, "
            f"{self.skipped} skipped, {self.failed} failed. "
            f"Quotes: {self.quotes_validated} kept, {self.quotes_discarded} discarded. "
            f"Time: {self.duration_seconds:.1f}s"
        )


def build_enrichment_prompt(core_text: str, case_context: Optional[CaseContext] = None) -> str:
    """
    Build the enrichment prompt for a single chunk.

    Shared across all backends (Ollama, Anthropic, Claude Code).

    Args:
        core_text: The chunk text to enrich
        case_context: Optional case context for better classification

    Returns:
        Prompt string ready to send to an LLM
    """
    context_section = ""
    if case_context:
        context_section = f"\n\nCase Context:\n{case_context.to_prompt_section()}\n"

    return f"""Analyze this litigation document excerpt and provide enrichment metadata.
{context_section}
Document text:
---
{core_text}
---

Return a JSON object with these fields:
- "summary": 2-3 sentences describing the legal significance of this text
- "key_quotes": array of 1-3 exact verbatim quotes from the text above (copy-paste exactly, do not paraphrase)
- "category": one of: legal_argument, factual_allegation, evidence_reference, expert_opinion, witness_statement, damages_analysis, procedural, background, statutory_regulatory, case_citation, claim_construction, infringement_analysis, validity_invalidity, unclassified
- "relevance_score": one of: high, medium, low
- "claims_addressed": array of integer patent claim numbers mentioned or relevant (empty array if none)

IMPORTANT: key_quotes must be EXACT substrings copied verbatim from the document text. Do not modify, paraphrase, or truncate them.

Return ONLY the JSON object, no other text."""


def validate_enrichment(
    enrichment: Dict[str, Any],
    core_text: str,
) -> Dict[str, Any]:
    """
    Validate and clean enrichment data per TRD Section 9.4.

    - Key quotes must be exact substrings of core_text (hallucinated ones discarded)
    - Category must be from VALID_CATEGORIES (default: unclassified)
    - Relevance must be high/medium/low (default: medium)
    - Claims must be integers < 100 (filter out patent numbers)

    Args:
        enrichment: Raw enrichment dict from LLM
        core_text: Original chunk text for quote verification

    Returns:
        Validated enrichment dict
    """
    validated = {}

    # Summary: keep as-is if it's a string
    summary = enrichment.get("summary")
    if isinstance(summary, str) and summary.strip():
        validated["summary"] = summary.strip()

    # Key quotes: must be exact substrings of core_text
    raw_quotes = enrichment.get("key_quotes", [])
    if isinstance(raw_quotes, list):
        valid_quotes = []
        for quote in raw_quotes:
            if isinstance(quote, str) and quote.strip() and quote.strip() in core_text:
                valid_quotes.append(quote.strip())
        validated["key_quotes"] = valid_quotes
    else:
        validated["key_quotes"] = []

    # Category: must be from allowed set
    category = enrichment.get("category", "unclassified")
    if isinstance(category, str) and category.strip().lower() in VALID_CATEGORIES:
        validated["category"] = category.strip().lower()
    else:
        validated["category"] = "unclassified"

    # Relevance score: must be high/medium/low
    relevance = enrichment.get("relevance_score", "medium")
    if isinstance(relevance, str) and relevance.strip().lower() in VALID_RELEVANCE:
        validated["relevance_score"] = relevance.strip().lower()
    else:
        validated["relevance_score"] = "medium"

    # Claims: must be integers < 100 (filter out patent numbers like 7654321)
    raw_claims = enrichment.get("claims_addressed", [])
    if isinstance(raw_claims, list):
        valid_claims = []
        for claim in raw_claims:
            if isinstance(claim, int) and 0 < claim < 100:
                valid_claims.append(claim)
            elif isinstance(claim, (str, float)):
                try:
                    val = int(claim)
                    if 0 < val < 100:
                        valid_claims.append(val)
                except (ValueError, TypeError):
                    pass
        validated["claims_addressed"] = sorted(set(valid_claims))
    else:
        validated["claims_addressed"] = []

    return validated


def _parse_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response text into a dict.

    Handles:
    - Clean JSON
    - Markdown-fenced JSON (```json ... ```)
    - Malformed JSON (best-effort)

    Returns:
        Parsed dict, or None if parsing fails
    """
    if not response_text or not isinstance(response_text, str):
        return None

    text = response_text.strip()

    # Strip markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try to find JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON")
        return None


class LLMEnricher:
    """
    Main enrichment class supporting Ollama, Anthropic, and Claude Code backends.

    Usage:
        enricher = LLMEnricher(backend="ollama")
        if enricher.is_available():
            stats = enricher.enrich_chunks_file("chunks.json", case_context)
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        delay_between_calls: float = 0.1,
        max_retries: int = 3,
    ):
        self.backend = backend
        self.ollama_url = ollama_url
        self.delay_between_calls = delay_between_calls
        self.max_retries = max_retries

        if model:
            self.model = model
        elif backend == "ollama":
            self.model = DEFAULT_OLLAMA_MODEL
        elif backend == "anthropic":
            self.model = DEFAULT_ANTHROPIC_MODEL
        else:
            self.model = None

        self._anthropic_client = None
        self._availability_checked = False
        self._available = False

    def is_available(self) -> bool:
        """Check if the configured backend is accessible."""
        if self._availability_checked:
            return self._available

        self._availability_checked = True

        if self.backend == "ollama":
            self._available = self._check_ollama()
        elif self.backend == "anthropic":
            self._available = self._check_anthropic()
        elif self.backend == "claude_code":
            self._available = True  # Always available in Claude Code context
        else:
            logger.warning(f"Unknown backend: {self.backend}")
            self._available = False

        return self._available

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and accessible."""
        import urllib.request
        import urllib.error

        try:
            # Check Ollama API root
            base_url = self.ollama_url.rsplit('/api/', 1)[0]
            req = urllib.request.Request(f"{base_url}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            logger.warning("Ollama not accessible at %s", self.ollama_url)
            return False

    def _check_anthropic(self) -> bool:
        """Check if Anthropic SDK is available and API key is set."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set")
            return False
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic()
            return True
        except ImportError:
            logger.warning(
                "anthropic SDK not installed. "
                "Install with: pip install anthropic"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")
            return False

    def enrich_chunks_file(
        self,
        chunks_path: str,
        case_context: Optional[CaseContext] = None,
        force: bool = False,
    ) -> EnrichmentStats:
        """
        Enrich all chunks in a *_chunks.json file.

        Creates a backup (*_chunks_pre_enrichment.json) on first run.
        Writes enrichment data back to the same file.

        Args:
            chunks_path: Path to chunks JSON file
            case_context: Optional case context
            force: Re-enrich already-enriched chunks

        Returns:
            EnrichmentStats with run metrics
        """
        path = Path(chunks_path)
        stats = EnrichmentStats()
        start_time = time.time()

        # Load chunks
        with open(path, 'r') as f:
            chunks_data = json.load(f)

        stats.total = len(chunks_data)

        # Create backup on first enrichment run
        backup_path = path.with_name(path.stem + "_pre_enrichment.json")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            logger.info("Backup created: %s", backup_path.name)

        # Process each chunk
        disable_progress = _should_disable_tqdm()
        pbar = tqdm(chunks_data, desc=f"  Chunks in {Path(path).name[:20]}",
                    unit="chunk", disable=disable_progress, position=1, leave=False)

        for chunk_data in pbar:
            chunk_id = chunk_data.get("chunk_id", "unknown")

            # Skip already-enriched unless force=True
            if not force and chunk_data.get("classification_method"):
                stats.skipped += 1
                continue

            core_text = chunk_data.get("core_text", "")
            if not core_text.strip():
                stats.skipped += 1
                continue

            # Call LLM
            result = self._enrich_single_chunk(core_text, case_context)

            if result.error:
                logger.warning("Failed to enrich %s: %s", chunk_id, result.error)
                stats.failed += 1
                continue

            # Validate enrichment
            raw_enrichment = {
                "summary": result.summary,
                "key_quotes": result.key_quotes,
                "category": result.category,
                "relevance_score": result.relevance_score,
                "claims_addressed": result.claims_addressed,
            }
            validated = validate_enrichment(raw_enrichment, core_text)

            # Track quote stats
            raw_count = len(result.key_quotes) if result.key_quotes else 0
            valid_count = len(validated.get("key_quotes", []))
            stats.quotes_validated += valid_count
            stats.quotes_discarded += raw_count - valid_count

            # Merge validated enrichment into chunk data
            chunk_data["summary"] = validated.get("summary")
            chunk_data["category"] = validated["category"]
            chunk_data["relevance_score"] = validated["relevance_score"]
            chunk_data["claims_addressed"] = validated["claims_addressed"]
            chunk_data["classification_method"] = "llm"
            chunk_data["llm_backend"] = self.backend

            # Merge LLM-validated quotes with existing deterministic quotes
            existing_quotes = chunk_data.get("key_quotes", [])
            llm_quotes = validated.get("key_quotes", [])
            merged = list(dict.fromkeys(existing_quotes + llm_quotes))
            chunk_data["key_quotes"] = merged

            stats.enriched += 1
            pbar.set_postfix_str(f"enriched={stats.enriched}, skipped={stats.skipped}")

            # Rate limiting
            if self.delay_between_calls > 0:
                time.sleep(self.delay_between_calls)

        # Write back atomically (write to temp, then rename)
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        temp_path.replace(path)

        stats.duration_seconds = time.time() - start_time
        logger.info(stats.summary())
        return stats

    def enrich_directory(
        self,
        directory: str,
        case_context: Optional[CaseContext] = None,
        force: bool = False,
    ) -> EnrichmentStats:
        """
        Enrich all *_chunks.json files in a directory.

        Args:
            directory: Directory containing chunk files
            case_context: Optional case context
            force: Re-enrich already-enriched chunks

        Returns:
            Aggregate EnrichmentStats
        """
        dir_path = Path(directory)
        chunk_files = sorted(dir_path.glob("*_chunks.json"))

        if not chunk_files:
            logger.warning("No chunk files found in %s", directory)
            return EnrichmentStats()

        aggregate = EnrichmentStats()
        start_time = time.time()

        disable_progress = _should_disable_tqdm()
        pbar = tqdm(chunk_files, desc="Enriching files", unit="file", disable=disable_progress)

        for chunk_file in pbar:
            pbar.set_postfix_str(f"{chunk_file.name[:25]}...")
            logger.info("Enriching: %s", chunk_file.name)
            stats = self.enrich_chunks_file(str(chunk_file), case_context, force=force)
            aggregate.total += stats.total
            aggregate.enriched += stats.enriched
            aggregate.skipped += stats.skipped
            aggregate.failed += stats.failed
            aggregate.quotes_validated += stats.quotes_validated
            aggregate.quotes_discarded += stats.quotes_discarded

        aggregate.duration_seconds = time.time() - start_time
        return aggregate

    def _enrich_single_chunk(
        self,
        core_text: str,
        case_context: Optional[CaseContext] = None,
    ) -> EnrichmentResult:
        """
        Send one chunk to the LLM and get enrichment back.

        Args:
            core_text: Chunk text to enrich
            case_context: Optional case context

        Returns:
            EnrichmentResult (check .error for failures)
        """
        prompt = build_enrichment_prompt(core_text, case_context)

        if self.backend == "ollama":
            response_text = self._call_ollama(prompt)
        elif self.backend == "anthropic":
            response_text = self._call_anthropic(prompt)
        else:
            return EnrichmentResult(
                chunk_id="",
                error=f"Unsupported backend for direct calls: {self.backend}"
            )

        if response_text is None:
            return EnrichmentResult(chunk_id="", error="No response from LLM")

        parsed = _parse_llm_response(response_text)
        if parsed is None:
            return EnrichmentResult(chunk_id="", error="Failed to parse LLM response")

        return EnrichmentResult(
            chunk_id="",
            summary=parsed.get("summary"),
            key_quotes=parsed.get("key_quotes", []),
            category=parsed.get("category", "unclassified"),
            relevance_score=parsed.get("relevance_score", "medium"),
            claims_addressed=parsed.get("claims_addressed", []),
        )

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Call Ollama via urllib.request (stdlib, no extra dependency).

        Uses format: "json" for structured output.

        Returns:
            Response text, or None on failure
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.URLError as e:
            logger.error("Ollama request failed: %s", e)
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Ollama response error: %s", e)
            return None

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """
        Call Anthropic API with exponential backoff retry.

        Returns:
            Response text, or None on failure
        """
        if self._anthropic_client is None:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic()
            except (ImportError, Exception) as e:
                logger.error("Failed to initialize Anthropic client: %s", e)
                return None

        for attempt in range(self.max_retries):
            try:
                message = self._anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Extract text from response
                if message.content and len(message.content) > 0:
                    return message.content[0].text
                return None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Anthropic API error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1, self.max_retries, e, wait
                    )
                    time.sleep(wait)
                else:
                    logger.error("Anthropic API failed after %d retries: %s", self.max_retries, e)
                    return None


# --- Claude Code support functions ---

def get_unenriched_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """
    Return list of chunks needing enrichment from a chunks file.

    For use by Claude Code Task agents to identify work.

    Args:
        chunks_path: Path to *_chunks.json file

    Returns:
        List of chunk dicts that lack classification_method
    """
    with open(chunks_path, 'r') as f:
        chunks_data = json.load(f)

    return [
        c for c in chunks_data
        if not c.get("classification_method") and c.get("core_text", "").strip()
    ]


def apply_enrichment(
    chunks_path: str,
    chunk_id: str,
    enrichment: Dict[str, Any],
    backend: str = "claude_code",
) -> bool:
    """
    Validate and merge enrichment for one chunk, then save.

    For use by Claude Code Task agents after analyzing chunk text.

    Args:
        chunks_path: Path to *_chunks.json file
        chunk_id: ID of chunk to enrich
        enrichment: Raw enrichment dict from the agent
        backend: Backend identifier for tracking

    Returns:
        True if successfully applied, False otherwise
    """
    path = Path(chunks_path)

    with open(path, 'r') as f:
        chunks_data = json.load(f)

    # Create backup on first enrichment
    backup_path = path.with_name(path.stem + "_pre_enrichment.json")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    # Find the target chunk
    target = None
    for chunk_data in chunks_data:
        if chunk_data.get("chunk_id") == chunk_id:
            target = chunk_data
            break

    if target is None:
        logger.warning("Chunk not found: %s", chunk_id)
        return False

    core_text = target.get("core_text", "")
    validated = validate_enrichment(enrichment, core_text)

    # Apply validated enrichment
    target["summary"] = validated.get("summary")
    target["category"] = validated["category"]
    target["relevance_score"] = validated["relevance_score"]
    target["claims_addressed"] = validated["claims_addressed"]
    target["classification_method"] = "llm"
    target["llm_backend"] = backend

    # Merge quotes
    existing_quotes = target.get("key_quotes", [])
    llm_quotes = validated.get("key_quotes", [])
    target["key_quotes"] = list(dict.fromkeys(existing_quotes + llm_quotes))

    # Write back atomically (write to temp, then rename)
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(chunks_data, f, indent=2)
    temp_path.replace(path)

    return True
