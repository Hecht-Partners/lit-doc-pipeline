"""
Tests for LLM enrichment module.

All tests are mock-based — no real API calls to Ollama or Anthropic.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lit_pipeline.llm_enrichment import (
    VALID_CATEGORIES,
    VALID_RELEVANCE,
    CaseContext,
    EnrichmentResult,
    EnrichmentStats,
    LLMEnricher,
    build_enrichment_prompt,
    validate_enrichment,
    get_unenriched_chunks,
    apply_enrichment,
    _parse_llm_response,
)


# --- Fixtures ---

SAMPLE_CORE_TEXT = (
    "Q. Can you describe the TWT technology used in the accused products?\n"
    "A. Yes, Target Wake Time is a power-saving mechanism defined in the "
    "IEEE 802.11ax standard. It allows devices to negotiate specific wake "
    "times with the access point, reducing unnecessary wake-ups and "
    "significantly improving battery life."
)

SAMPLE_CHUNK = {
    "chunk_id": "doc1_chunk_0001",
    "core_text": SAMPLE_CORE_TEXT,
    "pages": [14, 15],
    "citation": {"pdf_pages": [14, 15], "transcript_lines": {"14": [5, 25]}},
    "citation_string": "Alexander Dep. 14:5-15:12",
    "key_quotes": ["Target Wake Time is a power-saving mechanism"],
    "tokens": 62,
    "doc_type": "deposition",
}

SAMPLE_ENRICHED_CHUNK = {
    **SAMPLE_CHUNK,
    "summary": "Witness describes TWT technology.",
    "category": "witness_statement",
    "relevance_score": "high",
    "claims_addressed": [1, 7],
    "classification_method": "llm",
    "llm_backend": "ollama",
}


def _make_chunks_file(chunks_data, tmpdir):
    """Helper to create a temporary chunks file."""
    path = Path(tmpdir) / "doc1_chunks.json"
    with open(path, 'w') as f:
        json.dump(chunks_data, f)
    return str(path)


# --- Validation tests ---

class TestValidateEnrichment:

    def test_valid_quote_kept(self):
        """Exact substring quote is kept."""
        enrichment = {
            "key_quotes": ["Target Wake Time is a power-saving mechanism"],
            "category": "witness_statement",
            "relevance_score": "high",
        }
        result = validate_enrichment(enrichment, SAMPLE_CORE_TEXT)
        assert len(result["key_quotes"]) == 1
        assert "Target Wake Time" in result["key_quotes"][0]

    def test_hallucinated_quote_discarded(self):
        """Quote not in core_text is discarded."""
        enrichment = {
            "key_quotes": ["This quote does not exist in the text at all"],
            "category": "witness_statement",
            "relevance_score": "high",
        }
        result = validate_enrichment(enrichment, SAMPLE_CORE_TEXT)
        assert len(result["key_quotes"]) == 0

    def test_mixed_quotes_filtered(self):
        """Valid quotes kept, hallucinated ones discarded."""
        enrichment = {
            "key_quotes": [
                "Target Wake Time is a power-saving mechanism",
                "This is totally made up",
                "improving battery life",
            ],
            "category": "witness_statement",
            "relevance_score": "medium",
        }
        result = validate_enrichment(enrichment, SAMPLE_CORE_TEXT)
        assert len(result["key_quotes"]) == 2

    def test_valid_category_passes(self):
        """Valid category is accepted."""
        for cat in VALID_CATEGORIES:
            result = validate_enrichment({"category": cat}, "text")
            assert result["category"] == cat

    def test_invalid_category_defaults_unclassified(self):
        """Invalid category defaults to unclassified."""
        result = validate_enrichment({"category": "bogus_category"}, "text")
        assert result["category"] == "unclassified"

    def test_valid_relevance_passes(self):
        """Valid relevance scores are accepted."""
        for rel in VALID_RELEVANCE:
            result = validate_enrichment({"relevance_score": rel}, "text")
            assert result["relevance_score"] == rel

    def test_invalid_relevance_defaults_medium(self):
        """Invalid relevance defaults to medium."""
        result = validate_enrichment({"relevance_score": "critical"}, "text")
        assert result["relevance_score"] == "medium"

    def test_claim_integers_kept(self):
        """Integer claims < 100 are kept."""
        enrichment = {"claims_addressed": [1, 7, 15]}
        result = validate_enrichment(enrichment, "text")
        assert result["claims_addressed"] == [1, 7, 15]

    def test_patent_numbers_rejected(self):
        """Large numbers (patent numbers) are filtered out."""
        enrichment = {"claims_addressed": [1, 7654321, 7, 9876543]}
        result = validate_enrichment(enrichment, "text")
        assert result["claims_addressed"] == [1, 7]

    def test_string_claims_coerced(self):
        """String claim numbers are coerced to int."""
        enrichment = {"claims_addressed": ["1", "7", "not_a_number"]}
        result = validate_enrichment(enrichment, "text")
        assert result["claims_addressed"] == [1, 7]

    def test_full_validation(self):
        """End-to-end validation with all fields."""
        enrichment = {
            "summary": "Witness explains TWT power savings.",
            "key_quotes": [
                "Target Wake Time is a power-saving mechanism",
                "Hallucinated quote here",
            ],
            "category": "witness_statement",
            "relevance_score": "high",
            "claims_addressed": [1, 7, 9999999],
        }
        result = validate_enrichment(enrichment, SAMPLE_CORE_TEXT)
        assert result["summary"] == "Witness explains TWT power savings."
        assert len(result["key_quotes"]) == 1
        assert result["category"] == "witness_statement"
        assert result["relevance_score"] == "high"
        assert result["claims_addressed"] == [1, 7]


# --- Parsing tests ---

class TestParseLLMResponse:

    def test_clean_json(self):
        """Parse clean JSON string."""
        raw = '{"summary": "test", "category": "background"}'
        result = _parse_llm_response(raw)
        assert result["summary"] == "test"
        assert result["category"] == "background"

    def test_markdown_fenced_json(self):
        """Parse JSON wrapped in markdown code fences."""
        raw = '```json\n{"summary": "test", "category": "background"}\n```'
        result = _parse_llm_response(raw)
        assert result["summary"] == "test"

    def test_malformed_json(self):
        """Malformed JSON returns None."""
        raw = "This is not JSON at all {broken"
        result = _parse_llm_response(raw)
        assert result is None

    def test_empty_input(self):
        """Empty input returns None."""
        assert _parse_llm_response("") is None
        assert _parse_llm_response(None) is None

    def test_json_with_surrounding_text(self):
        """JSON embedded in surrounding text is extracted."""
        raw = 'Here is the analysis:\n{"summary": "test", "category": "background"}\nDone.'
        result = _parse_llm_response(raw)
        assert result["summary"] == "test"


# --- Prompt tests ---

class TestBuildPrompt:

    def test_basic_prompt(self):
        """Prompt includes core_text and expected fields."""
        prompt = build_enrichment_prompt("Some legal text")
        assert "Some legal text" in prompt
        assert "key_quotes" in prompt
        assert "category" in prompt
        assert "relevance_score" in prompt
        assert "claims_addressed" in prompt

    def test_prompt_with_case_context(self):
        """Case context is included in prompt."""
        ctx = CaseContext(
            case_type="patent",
            parties=["Proxim", "Intel"],
            patents=["'152 Patent"],
            claims_at_issue=[1, 7],
        )
        prompt = build_enrichment_prompt("Text", ctx)
        assert "patent" in prompt
        assert "Proxim" in prompt
        assert "'152 Patent" in prompt


# --- Integration tests (mocked) ---

class TestLLMEnricherOllama:

    def test_enrich_single_chunk_ollama(self):
        """Enrich a chunk with mocked Ollama response."""
        enricher = LLMEnricher(backend="ollama")

        mock_response = json.dumps({
            "summary": "Witness describes TWT technology.",
            "key_quotes": ["Target Wake Time is a power-saving mechanism"],
            "category": "witness_statement",
            "relevance_score": "high",
            "claims_addressed": [1, 7],
        })

        with patch.object(enricher, '_call_ollama', return_value=mock_response):
            result = enricher._enrich_single_chunk(SAMPLE_CORE_TEXT)

        assert result.success
        assert result.summary == "Witness describes TWT technology."
        assert result.category == "witness_statement"

    def test_enrich_chunks_file_writes_back(self):
        """Enrichment data is written back to the chunks file."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_CHUNK.copy()], tmpdir)
            enricher = LLMEnricher(backend="ollama", delay_between_calls=0)

            mock_response = json.dumps({
                "summary": "TWT testimony.",
                "key_quotes": ["Target Wake Time is a power-saving mechanism"],
                "category": "witness_statement",
                "relevance_score": "high",
                "claims_addressed": [1],
            })

            with patch.object(enricher, '_call_ollama', return_value=mock_response):
                stats = enricher.enrich_chunks_file(chunks_path)

            assert stats.enriched == 1
            assert stats.failed == 0

            # Verify file was updated
            with open(chunks_path) as f:
                updated = json.load(f)
            assert updated[0]["category"] == "witness_statement"
            assert updated[0]["classification_method"] == "llm"
            assert updated[0]["llm_backend"] == "ollama"
        finally:
            shutil.rmtree(tmpdir)

    def test_skip_already_enriched(self):
        """Already-enriched chunks are skipped."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_ENRICHED_CHUNK.copy()], tmpdir)
            enricher = LLMEnricher(backend="ollama", delay_between_calls=0)

            with patch.object(enricher, '_call_ollama') as mock_call:
                stats = enricher.enrich_chunks_file(chunks_path)

            assert stats.skipped == 1
            assert stats.enriched == 0
            mock_call.assert_not_called()
        finally:
            shutil.rmtree(tmpdir)

    def test_force_re_enrichment(self):
        """force=True re-enriches already-enriched chunks."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_ENRICHED_CHUNK.copy()], tmpdir)
            enricher = LLMEnricher(backend="ollama", delay_between_calls=0)

            mock_response = json.dumps({
                "summary": "Updated summary.",
                "key_quotes": [],
                "category": "expert_opinion",
                "relevance_score": "medium",
                "claims_addressed": [],
            })

            with patch.object(enricher, '_call_ollama', return_value=mock_response):
                stats = enricher.enrich_chunks_file(chunks_path, force=True)

            assert stats.enriched == 1
            assert stats.skipped == 0

            with open(chunks_path) as f:
                updated = json.load(f)
            assert updated[0]["category"] == "expert_opinion"
        finally:
            shutil.rmtree(tmpdir)

    def test_backup_file_created(self):
        """First enrichment creates a backup file."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_CHUNK.copy()], tmpdir)
            enricher = LLMEnricher(backend="ollama", delay_between_calls=0)

            mock_response = json.dumps({
                "summary": "Test.",
                "key_quotes": [],
                "category": "background",
                "relevance_score": "low",
                "claims_addressed": [],
            })

            with patch.object(enricher, '_call_ollama', return_value=mock_response):
                enricher.enrich_chunks_file(chunks_path)

            backup = Path(chunks_path).with_name("doc1_chunks_pre_enrichment.json")
            assert backup.exists()

            # Verify backup contains original data
            with open(backup) as f:
                backup_data = json.load(f)
            assert "classification_method" not in backup_data[0]
        finally:
            shutil.rmtree(tmpdir)

    def test_api_error_graceful(self):
        """API errors are handled gracefully."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_CHUNK.copy()], tmpdir)
            enricher = LLMEnricher(backend="ollama", delay_between_calls=0)

            with patch.object(enricher, '_call_ollama', return_value=None):
                stats = enricher.enrich_chunks_file(chunks_path)

            assert stats.failed == 1
            assert stats.enriched == 0
        finally:
            shutil.rmtree(tmpdir)


class TestLLMEnricherAnthropic:

    def test_enrich_single_chunk_anthropic(self):
        """Enrich a chunk with mocked Anthropic response."""
        enricher = LLMEnricher(backend="anthropic")

        mock_response = json.dumps({
            "summary": "Deponent explains power savings.",
            "key_quotes": ["improving battery life"],
            "category": "witness_statement",
            "relevance_score": "high",
            "claims_addressed": [1],
        })

        with patch.object(enricher, '_call_anthropic', return_value=mock_response):
            result = enricher._enrich_single_chunk(SAMPLE_CORE_TEXT)

        assert result.success
        assert result.category == "witness_statement"


# --- Claude Code support tests ---

class TestClaudeCodeSupport:

    def test_get_unenriched_chunks(self):
        """Returns only chunks without classification_method."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks = [SAMPLE_CHUNK.copy(), SAMPLE_ENRICHED_CHUNK.copy()]
            chunks_path = _make_chunks_file(chunks, tmpdir)
            unenriched = get_unenriched_chunks(chunks_path)
            assert len(unenriched) == 1
            assert unenriched[0]["chunk_id"] == "doc1_chunk_0001"
        finally:
            shutil.rmtree(tmpdir)

    def test_apply_enrichment_validates_and_saves(self):
        """apply_enrichment validates quotes and saves."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_CHUNK.copy()], tmpdir)
            enrichment = {
                "summary": "Test summary.",
                "key_quotes": [
                    "Target Wake Time is a power-saving mechanism",
                    "Hallucinated quote",
                ],
                "category": "witness_statement",
                "relevance_score": "high",
                "claims_addressed": [1, 7],
            }

            result = apply_enrichment(chunks_path, "doc1_chunk_0001", enrichment)
            assert result is True

            with open(chunks_path) as f:
                updated = json.load(f)

            chunk = updated[0]
            assert chunk["category"] == "witness_statement"
            assert chunk["classification_method"] == "llm"
            assert chunk["llm_backend"] == "claude_code"
            # Hallucinated quote should be filtered
            assert "Hallucinated quote" not in chunk["key_quotes"]
            # Valid quote should be present
            assert "Target Wake Time is a power-saving mechanism" in chunk["key_quotes"]
        finally:
            shutil.rmtree(tmpdir)

    def test_apply_enrichment_chunk_not_found(self):
        """Returns False when chunk_id not found."""
        tmpdir = tempfile.mkdtemp()
        try:
            chunks_path = _make_chunks_file([SAMPLE_CHUNK.copy()], tmpdir)
            result = apply_enrichment(chunks_path, "nonexistent_chunk", {})
            assert result is False
        finally:
            shutil.rmtree(tmpdir)


# --- Availability tests ---

class TestAvailability:

    def test_ollama_not_available(self):
        """Ollama unavailable when network fails."""
        enricher = LLMEnricher(backend="ollama")
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            assert enricher.is_available() is False

    def test_anthropic_no_key(self):
        """Anthropic unavailable without API key."""
        enricher = LLMEnricher(backend="anthropic")
        with patch.dict("os.environ", {}, clear=True):
            enricher._availability_checked = False
            assert enricher.is_available() is False

    def test_claude_code_always_available(self):
        """Claude Code backend is always available."""
        enricher = LLMEnricher(backend="claude_code")
        assert enricher.is_available() is True
