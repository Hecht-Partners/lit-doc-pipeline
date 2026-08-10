"""
Section-aware chunking for litigation documents.

Creates semantic chunks that preserve document structure and citation metadata:
- Depositions: Never split Q/A pairs
- Expert reports: Preserve paragraph boundaries with inline footnotes
- Patents: Preserve claim structure and column formatting
- All types: Attach complete citation metadata (page, Bates, line numbers)

Output: Context cards ready for vector indexing.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lit_pipeline.citation_types import DocumentType, Chunk

logger = logging.getLogger(__name__)

# Chunking parameters
DEFAULT_TARGET_TOKENS = 800  # Target chunk size
DEFAULT_MAX_TOKENS = 1200    # Hard limit before forcing split
DEFAULT_OVERLAP_TOKENS = 100  # Overlap between chunks
CHARS_PER_TOKEN = 4          # Rough estimate: 1 token ≈ 4 characters

# Filename Bates-range pattern: "ABBOTT0000001-0000012 ..." / "abbott0000001_0000012_..."
# Group 1 = prefix, 2 = start digits, 3 = end digits (optionally re-prefixed).
FILENAME_BATES_RE = re.compile(
    r"^([A-Za-z]{2,10})[ _-]?(\d{5,10})\s*[-_– ]\s*(?:[A-Za-z]{2,10}[ _-]?)?(\d{5,10})(?!\d)"
)


def format_page_ranges(pages) -> str:
    """Format a collection of page numbers as a precise citation fragment.

    Consecutive runs collapse to ranges; gaps are preserved instead of being
    swallowed into a misleading min-max span:
        [1, 6, 8, 9, 10] -> "pp. 1, 6, 8-10"
        [4]              -> "p. 4"
    """
    uniq = sorted({p for p in pages if p is not None})
    if not uniq:
        return ""
    runs = []
    start = prev = uniq[0]
    for p in uniq[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev))
        start = prev = p
    runs.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in runs]
    label = "p." if len(uniq) == 1 else "pp."
    return f"{label} {', '.join(parts)}"


def compute_page_spans(page_map: List[Optional[int]]) -> List[dict]:
    """Compress a per-line page_map into contiguous line-range spans.

    Lines are 1-indexed into core_text; lines with unknown page (None) are
    omitted. Example: [1, 1, 6, 6, 1] ->
        [{"page": 1, "line_start": 1, "line_end": 2},
         {"page": 6, "line_start": 3, "line_end": 4},
         {"page": 1, "line_start": 5, "line_end": 5}]
    """
    spans: List[dict] = []
    for i, page in enumerate(page_map, start=1):
        if page is None:
            continue
        if spans and spans[-1]["page"] == page and spans[-1]["line_end"] == i - 1:
            spans[-1]["line_end"] = i
        else:
            spans.append({"page": page, "line_start": i, "line_end": i})
    return spans


@dataclass
class ChunkMetadata:
    """Metadata collected while building a chunk."""
    text_ids: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    bates_stamps: List[str] = field(default_factory=list)
    line_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # page -> (line_start, line_end)
    paragraph_numbers: List[int] = field(default_factory=list)
    columns: List[int] = field(default_factory=list)
    transcript_pages: List[int] = field(default_factory=list)
    # Discovery-response: which numbered request this chunk answers
    discovery_request_kind: Optional[str] = None       # "rog" | "rfp" | "rfa"
    discovery_request_number: Optional[int] = None     # e.g., 21
    discovery_part: Optional[int] = None               # 1-indexed for split chunks
    discovery_part_total: Optional[int] = None         # total parts when split


class DocumentChunker:
    """
    Create semantic chunks from processed markdown with citation metadata.

    Reads:
    - {stem}.md: Markdown with [TEXT:N], [PAGE:N], [FOOTNOTE:...] markers
    - {stem}_citations.json: Citation metadata keyed by #/texts/N or line_P*_L*

    Outputs:
    - {stem}_chunks.json: Array of context cards with full citation data
    """

    def __init__(
        self,
        converted_dir: str,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.converted_dir = Path(converted_dir)
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # Type sets for handler routing
    TRANSCRIPT_TYPES = {DocumentType.DEPOSITION, DocumentType.HEARING_TRANSCRIPT}
    PARAGRAPH_TYPES = {
        DocumentType.EXPERT_REPORT, DocumentType.PLEADING,
        DocumentType.DECLARATION, DocumentType.MOTION,
        DocumentType.BRIEF, DocumentType.WITNESS_STATEMENT,
        DocumentType.AGREEMENT,
    }
    PATENT_TYPES = {DocumentType.PATENT}
    DISCOVERY_TYPES = {DocumentType.DISCOVERY_RESPONSE, DocumentType.DISCOVERY_REQUEST}

    def chunk_document(
        self,
        stem: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        source_file: str = "",
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Create chunks from a processed document.

        Args:
            stem: Document stem (e.g., "daniel_alexander_10_24_2025")
            doc_type: Document type for type-specific handling
            source_file: Original PDF filename for metadata

        Returns:
            List of Chunk objects with complete citation data
        """
        # Load inputs
        md_content, citations = self._load_inputs(stem)
        if not md_content or not citations:
            logger.warning("Missing inputs for %s", stem)
            return []

        # Productions often carry no extractable Bates in page footers but
        # encode the range in the filename; derive one stamp per page so
        # chunks get precise Bates pinpoints (no-op when Bates already exist
        # or the filename range doesn't match the page count).
        self.derive_bates_by_page(source_file or stem, citations)

        # Parse markdown into sections
        sections = self._parse_markdown(md_content, doc_type)
        logger.info("Parsed %d sections from markdown", len(sections))

        # Create chunks from sections - route by type sets
        chunks = []
        if doc_type in self.TRANSCRIPT_TYPES:
            chunks = self._chunk_deposition(sections, citations, stem, source_file, source_path=source_path)
        elif doc_type in self.DISCOVERY_TYPES:
            chunks = self._chunk_discovery(sections, citations, stem, source_file, doc_type, source_path=source_path)
        elif doc_type in self.PARAGRAPH_TYPES:
            chunks = self._chunk_expert_report(sections, citations, stem, source_file, source_path=source_path)
        elif doc_type in self.PATENT_TYPES:
            chunks = self._chunk_patent(sections, citations, stem, source_file, source_path=source_path)
        else:
            chunks = self._chunk_generic(sections, citations, stem, source_file, source_path=source_path)

        logger.info("Created %d chunks from %s", len(chunks), stem)

        # Save chunks
        output_path = self.converted_dir / f"{stem}_chunks.json"
        with open(output_path, "w") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)

        logger.info("Saved chunks to %s", output_path)
        return chunks

    # ── Loading ──────────────────────────────────────────────────────

    def _load_inputs(self, stem: str) -> Tuple[Optional[str], Optional[dict]]:
        """Load markdown and citations JSON."""
        md_path = self.converted_dir / f"{stem}.md"
        citations_path = self.converted_dir / f"{stem}_citations.json"

        if not md_path.exists():
            logger.error("Markdown file not found: %s", md_path)
            return None, None

        if not citations_path.exists():
            logger.error("Citations file not found: %s", citations_path)
            return None, None

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        with open(citations_path, "r", encoding="utf-8") as f:
            citations = json.load(f)

        return md_content, citations

    # ── Markdown Parsing ─────────────────────────────────────────────

    def _parse_markdown(self, content: str, doc_type: DocumentType) -> List[dict]:
        """
        Parse markdown into logical sections.

        Returns:
            List of section dicts with 'lines', 'text_markers', 'type'.
            Each entry in 'lines' is a tuple (line_text, text_id_or_None)
            where text_id is the most recent [TEXT:N] marker that applies
            to this line.
        """
        lines = content.split("\n")
        sections = []
        current_section = {
            "lines": [],
            "text_markers": [],
            "type": "content"
        }
        active_text_id = None  # Track the most recent [TEXT:N]

        for line in lines:
            # Detect section boundaries
            if line.startswith("## ") or line.startswith("# "):
                # Save current section and start new one
                if current_section["lines"]:
                    sections.append(current_section)
                current_section = {
                    "lines": [(line, active_text_id)],
                    "text_markers": [],
                    "type": "header"
                }
                continue

            # Track [TEXT:N] markers
            if line.startswith("[TEXT:"):
                match = re.match(r"\[TEXT:(\d+)\]", line)
                if match:
                    active_text_id = match.group(1)
                    current_section["text_markers"].append(active_text_id)
                # Don't add the marker line to content
                continue

            # Skip other markers (will be parsed when needed)
            if line.startswith("[PAGE:") or line.startswith("[BATES:"):
                current_section["lines"].append((line, active_text_id))
                continue

            current_section["lines"].append((line, active_text_id))

        # Add final section
        if current_section["lines"]:
            sections.append(current_section)

        return sections

    # ── Deposition Chunking ──────────────────────────────────────────

    def _chunk_deposition(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Chunk deposition preserving Q/A pairs.

        CRITICAL: Never split a Q from its A.
        """
        chunks = []
        current_chunk_lines = []
        # Per-line tracking: (page, bates) for each line in current_chunk_lines
        current_line_attrs = []
        current_metadata = ChunkMetadata()

        for section in sections:
            for line, _text_id in section["lines"]:
                # Skip table separator lines (|---|---|)
                if re.match(r'^\s*\|[\s\-]+\|', line):
                    continue

                # Track page markers
                if line.startswith("[PAGE:"):
                    match = re.match(r"\[PAGE:(\d+)\]", line)
                    if match:
                        page = int(match.group(1))
                        if page not in current_metadata.transcript_pages:
                            current_metadata.transcript_pages.append(page)
                    continue

                # Parse line number and Q/A marker (handle both standard and table formats)
                # Standard format: " 5    Q  Question text" or " 5    Q.  Question text"
                # Table format: "|  | Q. Question text  | timestamp |" or "Q.  Question text"
                match = re.match(r"^\s*(\d{1,2})\s+([QA])\.?\s+(.+)$", line)
                if not match:
                    # Try table format: "| Q. Text |" or just "Q.  Text"
                    table_match = re.search(r'\|\s*([QA])\.?\s+(.+?)\s*\|', line)
                    if table_match:
                        # No explicit line number in table - use text_id to look up citation
                        qa_marker = table_match.group(1)
                        text = table_match.group(2)
                        # Get line number from citation if available
                        if _text_id is not None:
                            text_id_key = f"#/texts/{_text_id}"
                            cit = citations.get(text_id_key)
                            if cit and cit.get("line_start"):
                                match = type('obj', (object,), {
                                    'group': lambda self, n: [None, str(cit.get("line_start")), qa_marker, text][n]
                                })()

                if match:
                    line_num = int(match.group(1))
                    qa_marker = match.group(2)
                    text = match.group(3)

                    # Look up citation for this line (text_id first if available, then PyMuPDF, then Docling fallback)
                    current_page = current_metadata.transcript_pages[-1] if current_metadata.transcript_pages else 1
                    line_page = current_page
                    line_bates = None

                    # Try text_id first (Docling format)
                    cit = None
                    if _text_id is not None:
                        text_id_key = f"#/texts/{_text_id}"
                        cit = citations.get(text_id_key)

                    # Fallback to page/line lookup if text_id didn't work
                    if not cit:
                        cit = self._find_deposition_citation(citations, current_page, line_num)
                    if cit:
                        page = cit.get("page", current_page)
                        line_page = page
                        if page not in current_metadata.pages:
                            current_metadata.pages.append(page)

                        # Track line range for this page (use 'page' not 'current_page' - Bug Fix #1)
                        if page not in current_metadata.line_ranges:
                            current_metadata.line_ranges[page] = (line_num, line_num)
                        else:
                            start, end = current_metadata.line_ranges[page]
                            current_metadata.line_ranges[page] = (min(start, line_num), max(end, line_num))

                        bates = cit.get("bates")
                        line_bates = bates
                        if bates and bates not in current_metadata.bates_stamps:
                            current_metadata.bates_stamps.append(bates)

                    # Clean table formatting before adding to chunk
                    # Table format: "| text | timestamp |" → "text"
                    clean_line = line
                    if '|' in line:
                        # Extract content from table cells (remove pipes and timestamps)
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        # Remove timestamp (09:30:50 format)
                        parts = [p for p in parts if not re.match(r'^\d{2}:\d{2}:\d{2}', p)]
                        if parts:
                            # Reconstruct as "N Q text" or "N A text"
                            clean_line = f"{line_num} {qa_marker} {text}"

                    current_chunk_lines.append(clean_line)
                    current_line_attrs.append((line_page, line_bates))

                    # Check if we should start a new chunk
                    # Rule 1: If this is a Q, check if adding the expected A would exceed max_tokens
                    # Rule 2: Never split before an A
                    chunk_text = "\n".join(current_chunk_lines)
                    tokens = len(chunk_text) // CHARS_PER_TOKEN

                    if qa_marker == "A" and tokens >= self.target_tokens:
                        # Build per-line maps from tracked attrs
                        page_map = [a[0] for a in current_line_attrs]
                        bates_map = [a[1] for a in current_line_attrs]
                        # Complete chunk after this A
                        chunk = self._create_chunk(
                            chunk_text, current_metadata, stem, source_file,
                            len(chunks), DocumentType.DEPOSITION,
                            page_map=page_map, bates_map=bates_map,
                            source_path=source_path,
                        )
                        chunks.append(chunk)

                        # Start new chunk with overlap (last few lines) - Bug Fix #3
                        overlap_count = min(3, len(current_chunk_lines))
                        overlap_lines = current_chunk_lines[-overlap_count:] if overlap_count else []
                        overlap_attrs = current_line_attrs[-overlap_count:] if overlap_count else []
                        current_chunk_lines = overlap_lines
                        current_line_attrs = overlap_attrs

                        # Preserve FULL metadata for overlap lines, not just last page
                        current_metadata = ChunkMetadata()
                        overlap_pages = sorted(set(attr[0] for attr in overlap_attrs if attr[0]))
                        overlap_bates = [attr[1] for attr in overlap_attrs if attr[1]]

                        if overlap_pages:
                            current_metadata.pages = overlap_pages
                            current_metadata.transcript_pages = overlap_pages  # Assume transcript pages match for depositions
                        if overlap_bates:
                            current_metadata.bates_stamps = overlap_bates

                        # Preserve line_ranges for overlap lines from previous chunk
                        if hasattr(chunk, 'citation') and chunk.citation.get('transcript_lines'):
                            # Copy line ranges that appear in overlap
                            for page_str, line_range in chunk.citation['transcript_lines'].items():
                                page_num = int(page_str)
                                if page_num in overlap_pages:
                                    current_metadata.line_ranges[page_num] = tuple(line_range)

                else:
                    # Clean table formatting from non-Q/A lines too
                    clean_line = line
                    if '|' in line:
                        # Remove pipes and timestamps
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        parts = [p for p in parts if not re.match(r'^\d{2}:\d{2}:\d{2}', p) and not re.match(r'^-+$', p)]
                        if parts:
                            clean_line = ' '.join(parts)
                        else:
                            # Skip pure separator/timestamp lines
                            continue

                    current_chunk_lines.append(clean_line)
                    # Non-Q/A lines: inherit last known page/bates
                    last_page = current_line_attrs[-1][0] if current_line_attrs else None
                    last_bates = current_line_attrs[-1][1] if current_line_attrs else None
                    current_line_attrs.append((last_page, last_bates))

        # Add final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            page_map = [a[0] for a in current_line_attrs]
            bates_map = [a[1] for a in current_line_attrs]
            chunk = self._create_chunk(
                chunk_text, current_metadata, stem, source_file,
                len(chunks), DocumentType.DEPOSITION,
                page_map=page_map, bates_map=bates_map,
                source_path=source_path,
            )
            chunks.append(chunk)

        return chunks

    def _find_deposition_citation(
        self,
        citations: dict,
        page: int,
        line_num: int,
    ) -> Optional[dict]:
        """Look up citation data for a deposition line.

        Tries the PyMuPDF key format first (line_P{page}_L{line}), then
        falls back to scanning all citations for a matching page + line_start
        (Docling #/texts/N key format).

        Returns the citation dict, or None if not found.
        """
        # Fast path: PyMuPDF key format
        pymupdf_key = f"line_P{page}_L{line_num}"
        if pymupdf_key in citations:
            return citations[pymupdf_key]

        # Fallback: scan for Docling-style keys with matching page + line_start
        for key, cit in citations.items():
            if (cit.get("transcript_page") == page
                    and cit.get("line_start") == line_num):
                return cit
            # Also check page field for non-transcript citations
            if (cit.get("page") == page
                    and cit.get("line_start") == line_num
                    and cit.get("type") == "transcript_line"):
                return cit

        return None

    # ── Expert Report Chunking ───────────────────────────────────────

    def _chunk_expert_report(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Chunk expert report by paragraphs with inline footnotes.

        Strategy: Preserve paragraph boundaries, include footnotes with their paragraphs.
        """
        chunks = []
        current_chunk_entries = []  # (line_text, text_id) tuples
        current_metadata = ChunkMetadata()

        for section in sections:
            for line_text, text_id in section["lines"]:
                # Skip empty lines
                if not line_text.strip():
                    current_chunk_entries.append((line_text, text_id))
                    continue

                # Detect paragraph start (enhanced patterns - Bug Fix #4)
                para_match = (
                    re.match(r"^¶\s*(\d+)", line_text) or           # ¶ 42
                    re.match(r"^§\s*(\d+)", line_text) or           # § 42
                    re.match(r"^Paragraph\s+(\d+)", line_text) or   # Paragraph 42
                    re.match(r"^(\d+)\.\s+[A-Z]", line_text)        # 42. I declare...
                )

                # Check if we should start a new chunk
                chunk_text = "\n".join(t for t, _ in current_chunk_entries)
                tokens = len(chunk_text) // CHARS_PER_TOKEN

                # Split at paragraph boundaries when target size reached (Bug Fix #4)
                if para_match and tokens >= self.target_tokens and current_chunk_entries:
                    page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
                    chunk = self._create_chunk(
                        chunk_text, current_metadata, stem, source_file,
                        len(chunks), DocumentType.EXPERT_REPORT,
                        page_map=page_map, bates_map=bates_map,
                        source_path=source_path,
                    )
                    chunks.append(chunk)

                    # Start new chunk (no overlap for expert reports - paragraphs are self-contained)
                    current_chunk_entries = []
                    current_metadata = ChunkMetadata()
                elif tokens >= self.max_tokens and current_chunk_entries:
                    # HARD limit - must split even if not at paragraph boundary
                    logger.warning(f"Chunk exceeds max_tokens ({self.max_tokens}), forcing split mid-paragraph")
                    page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
                    chunk = self._create_chunk(
                        chunk_text, current_metadata, stem, source_file,
                        len(chunks), DocumentType.EXPERT_REPORT,
                        page_map=page_map, bates_map=bates_map,
                        source_path=source_path,
                    )
                    chunks.append(chunk)
                    current_chunk_entries = []
                    current_metadata = ChunkMetadata()

                current_chunk_entries.append((line_text, text_id))

                # Apply citation metadata for this line's text_id
                if text_id and text_id not in current_metadata.text_ids:
                    current_metadata.text_ids.append(text_id)
                    cite_key = f"#/texts/{text_id}"
                    if cite_key in citations:
                        self._update_metadata(current_metadata, citations[cite_key])

        # Add final chunk
        if current_chunk_entries:
            chunk_text = "\n".join(t for t, _ in current_chunk_entries)
            page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
            chunk = self._create_chunk(
                chunk_text, current_metadata, stem, source_file,
                len(chunks), DocumentType.EXPERT_REPORT,
                page_map=page_map, bates_map=bates_map,
                source_path=source_path,
            )
            chunks.append(chunk)

        return chunks

    # ── Discovery (request/response) Chunking ────────────────────────

    # Boundary regex matches the START of a numbered request OR response.
    # Allows leading whitespace, optional markdown header prefix (`## `,
    # `### `), and optional bold-marker stars (`**`). Must mirror the pattern
    # used in citation_tracker.CitationTracker.DISCOVERY_BOUNDARY_RE so headers
    # and text-ids stay in sync.
    DISCOVERY_BOUNDARY_RE = re.compile(
        r"^\s*"
        r"(?:#+\s*)?"
        r"(?:\*+\s*)?"
        r"(?:RESPONSE\s+TO\s+|SUPPLEMENTAL\s+RESPONSE\s+TO\s+|FURTHER\s+RESPONSE\s+TO\s+)?"
        r"(INTERROGATORY|REQUEST\s+FOR\s+PRODUCTION|REQUEST\s+FOR\s+ADMISSION)"
        r"\s+NO\.\s*(\d+)",
        re.I,
    )

    @staticmethod
    def _discovery_kind_from_label(label: str) -> str:
        u = label.upper()
        if "PRODUCTION" in u:
            return "rfp"
        if "ADMISSION" in u:
            return "rfa"
        return "rog"

    def _chunk_discovery(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        doc_type: DocumentType,
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Chunk a discovery request or response by numbered-request boundaries.

        One chunk per (kind, number) bundle: every line under
        ``INTERROGATORY NO. 5`` (or RFP/RFA) — including the quoted request
        text, objections, response, and any supplemental responses — until the
        next numbered boundary.

        If a single bundle's token count exceeds ``self.max_tokens``, it is
        split with the request header repeated at the top of each split chunk
        and a small overlap between consecutive splits.
        """
        # Step 1: walk the sections and group line entries by (kind, number).
        # A "preamble" group (kind=None, number=None) collects everything
        # before the first numbered boundary (caption, general objections,
        # statement of definitions, etc.).
        groups: List[dict] = []
        current = {
            "kind": None,
            "number": None,
            "header_line": None,
            "entries": [],   # list of (line_text, text_id)
        }
        groups.append(current)

        for section in sections:
            for line_text, text_id in section["lines"]:
                stripped = line_text.strip()
                if not stripped:
                    current["entries"].append((line_text, text_id))
                    continue
                m = self.DISCOVERY_BOUNDARY_RE.match(stripped)
                # Only treat as a boundary if it's the FIRST occurrence for
                # this (kind, number) — i.e., the request header. The matching
                # "RESPONSE TO INTERROGATORY NO. N" line should not start a
                # new chunk; it lives inside the same bundle.
                is_request_header = bool(m) and not re.match(
                    r"^\s*(?:#+\s*)?(?:\*+\s*)?(?:RESPONSE\s+TO|SUPPLEMENTAL\s+RESPONSE\s+TO|FURTHER\s+RESPONSE\s+TO)\b",
                    stripped, re.I,
                )
                if is_request_header:
                    kind = self._discovery_kind_from_label(m.group(1))
                    number = int(m.group(2))
                    # Open a new group for this request number.
                    current = {
                        "kind": kind,
                        "number": number,
                        "header_line": line_text,
                        "entries": [(line_text, text_id)],
                    }
                    groups.append(current)
                else:
                    current["entries"].append((line_text, text_id))

        # Drop any preamble that's effectively empty (just blank lines).
        if groups and groups[0]["number"] is None:
            preamble_text = "\n".join(t for t, _ in groups[0]["entries"]).strip()
            if not preamble_text:
                groups.pop(0)

        # Step 2: build chunks per group. Split oversized bundles, repeating
        # the request header at the top of each split.
        chunks: List[Chunk] = []
        for g in groups:
            entries = g["entries"]
            if not entries:
                continue
            full_text = "\n".join(t for t, _ in entries)
            tokens = len(full_text) // CHARS_PER_TOKEN

            if tokens <= self.max_tokens:
                # Single chunk for this bundle (including preamble groups
                # where number is None).
                chunks.append(self._build_discovery_chunk(
                    entries, citations, stem, source_file, len(chunks),
                    doc_type=doc_type,
                    kind=g["kind"], number=g["number"],
                    part=None, part_total=None,
                    header_line=g["header_line"],
                    source_path=source_path,
                ))
                continue

            # Oversized — split with header repeat and overlap.
            splits = self._split_discovery_entries(
                entries,
                header_line=g["header_line"],
                target_tokens=self.target_tokens,
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
            )
            for part_idx, split_entries in enumerate(splits, start=1):
                chunks.append(self._build_discovery_chunk(
                    split_entries, citations, stem, source_file, len(chunks),
                    doc_type=doc_type,
                    kind=g["kind"], number=g["number"],
                    part=part_idx, part_total=len(splits),
                    header_line=g["header_line"],
                    source_path=source_path,
                ))

        return chunks

    def _split_discovery_entries(
        self,
        entries: List[Tuple[str, Optional[str]]],
        header_line: Optional[str],
        target_tokens: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> List[List[Tuple[str, Optional[str]]]]:
        """
        Split an oversized request bundle into chunks. Each non-first split
        gets the request header re-prepended (with a "(cont'd)" marker) so
        retrieval still surfaces which numbered request it belongs to.
        Consecutive splits share roughly ``overlap_tokens`` of overlap.
        """
        if not entries:
            return []

        # Lines already include the header_line as entries[0] (when present).
        # We accumulate to ~target_tokens, then close out the split.
        splits: List[List[Tuple[str, Optional[str]]]] = []
        cur: List[Tuple[str, Optional[str]]] = []
        cur_chars = 0
        target_chars = target_tokens * CHARS_PER_TOKEN
        max_chars = max_tokens * CHARS_PER_TOKEN
        overlap_chars = overlap_tokens * CHARS_PER_TOKEN

        for entry in entries:
            line_text, _tid = entry
            cur.append(entry)
            cur_chars += len(line_text) + 1  # +1 for the newline
            if cur_chars >= target_chars:
                splits.append(cur)
                # Build overlap tail: walk backwards collecting entries until
                # ~overlap_chars worth of text has been gathered.
                overlap_entries: List[Tuple[str, Optional[str]]] = []
                acc = 0
                for back in reversed(cur):
                    if acc >= overlap_chars:
                        break
                    overlap_entries.insert(0, back)
                    acc += len(back[0]) + 1
                # Start the next split with the header re-prepended (marked
                # cont'd) and then the overlap.
                next_split: List[Tuple[str, Optional[str]]] = []
                if header_line is not None:
                    next_split.append((f"{header_line}  (cont’d)", None))
                next_split.extend(overlap_entries)
                cur = next_split
                cur_chars = sum(len(t) + 1 for t, _ in cur)

        # Final split — only emit if it has more than just the repeated header.
        if cur:
            non_header_chars = sum(
                len(t) + 1 for t, _ in cur
                if not (header_line and t.startswith(header_line[:30]))
            )
            if non_header_chars > 0 or not splits:
                splits.append(cur)

        return splits

    def _build_discovery_chunk(
        self,
        entries: List[Tuple[str, Optional[str]]],
        citations: dict,
        stem: str,
        source_file: str,
        chunk_idx: int,
        doc_type: DocumentType,
        kind: Optional[str],
        number: Optional[int],
        part: Optional[int],
        part_total: Optional[int],
        header_line: Optional[str],
        source_path: str = "",
    ) -> Chunk:
        """Materialize a discovery chunk from a list of entries."""
        chunk_text = "\n".join(t for t, _ in entries)
        page_map, bates_map = self._build_line_maps(entries, citations)

        metadata = ChunkMetadata()
        for _line_text, text_id in entries:
            if text_id and text_id not in metadata.text_ids:
                metadata.text_ids.append(text_id)
                cite_key = f"#/texts/{text_id}"
                if cite_key in citations:
                    self._update_metadata(metadata, citations[cite_key])

        metadata.discovery_request_kind = kind
        metadata.discovery_request_number = number
        metadata.discovery_part = part
        metadata.discovery_part_total = part_total

        return self._create_chunk(
            chunk_text, metadata, stem, source_file, chunk_idx, doc_type,
            page_map=page_map, bates_map=bates_map,
            source_path=source_path,
        )

    # ── Patent Chunking ──────────────────────────────────────────────

    def _chunk_patent(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """Chunk patent preserving claim structure."""
        # For now, use generic chunking
        # TODO: Implement claim-aware chunking
        return self._chunk_generic(sections, citations, stem, source_file, source_path=source_path)

    # ── Generic Chunking ─────────────────────────────────────────────

    def _chunk_generic(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """Generic chunking by token count."""
        chunks = []
        # Each entry is (line_text, text_id_or_None)
        current_chunk_entries = []
        current_metadata = ChunkMetadata()

        for section in sections:
            for line_text, text_id in section["lines"]:
                # Apply citation metadata for this line's text_id
                if text_id and text_id not in current_metadata.text_ids:
                    current_metadata.text_ids.append(text_id)
                    cite_key = f"#/texts/{text_id}"
                    if cite_key in citations:
                        self._update_metadata(current_metadata, citations[cite_key])

                current_chunk_entries.append((line_text, text_id))

                # Check chunk size
                chunk_text = "\n".join(t for t, _ in current_chunk_entries)
                tokens = len(chunk_text) // CHARS_PER_TOKEN

                if tokens >= self.target_tokens:
                    page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
                    chunk = self._create_chunk(
                        chunk_text, current_metadata, stem, source_file,
                        len(chunks), DocumentType.UNKNOWN,
                        page_map=page_map, bates_map=bates_map,
                        source_path=source_path,
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_entries = current_chunk_entries[-5:] if len(current_chunk_entries) > 5 else []
                    current_chunk_entries = overlap_entries
                    current_metadata = ChunkMetadata()
                    # Re-apply metadata for overlap lines
                    for _, oid in overlap_entries:
                        if oid and oid not in current_metadata.text_ids:
                            current_metadata.text_ids.append(oid)
                            cite_key = f"#/texts/{oid}"
                            if cite_key in citations:
                                self._update_metadata(current_metadata, citations[cite_key])

        # Add final chunk
        if current_chunk_entries:
            chunk_text = "\n".join(t for t, _ in current_chunk_entries)
            page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
            chunk = self._create_chunk(
                chunk_text, current_metadata, stem, source_file,
                len(chunks), DocumentType.UNKNOWN,
                page_map=page_map, bates_map=bates_map,
                source_path=source_path,
            )
            chunks.append(chunk)

        return chunks

    # ── Page Precision Helpers ───────────────────────────────────────

    @staticmethod
    def _insert_page_markers(
        text: str,
        page_map: List[Optional[int]],
        bates_map: Optional[List[Optional[str]]],
    ) -> Tuple[str, List[Optional[int]], Optional[List[Optional[str]]]]:
        """Insert explicit [PAGE:N] marker lines at every page transition.

        The markers make pagination directly readable in core_text so a
        reviewing LLM never has to infer a quote's page from text position.
        Marker lines inherit the page (and Bates) of the line they precede,
        keeping page_map/bates_map aligned 1:1 with core_text lines.

        If page_map does not align with the text's lines, the input is
        returned unchanged (defensive: never corrupt alignment).
        """
        lines = text.split("\n")
        if len(lines) != len(page_map):
            logger.warning(
                "page_map length %d != line count %d; skipping page markers",
                len(page_map), len(lines),
            )
            return text, page_map, bates_map
        have_bates = bool(bates_map) and len(bates_map) == len(lines)

        out_lines: List[str] = []
        out_pages: List[Optional[int]] = []
        out_bates: List[Optional[str]] = []
        current_page: Optional[int] = None

        for i, line in enumerate(lines):
            page = page_map[i]
            bates = bates_map[i] if have_bates else None
            if page is not None and page != current_page:
                marker = f"[PAGE:{page} | {bates}]" if bates else f"[PAGE:{page}]"
                out_lines.append(marker)
                out_pages.append(page)
                out_bates.append(bates)
                current_page = page
            out_lines.append(line)
            out_pages.append(page)
            out_bates.append(bates)

        return (
            "\n".join(out_lines),
            out_pages,
            out_bates if have_bates else bates_map,
        )

    @staticmethod
    def derive_bates_by_page(
        source_name: str,
        citations: dict,
    ) -> int:
        """Derive per-page Bates stamps from a filename-encoded Bates range.

        Productions are conventionally named "{BEGBATES}-{ENDBATES} {title}"
        (e.g. "ABBOTT0000001-0000012 I3 DDD.pdf"). When no Bates stamps were
        extracted from page footers AND the filename range exactly matches the
        document's page count (one stamp per page), fill each citation's
        bates as prefix + (start + page - 1), preserving zero-padding.

        Mutates the citations dict in place. Returns the number of citation
        entries updated (0 when the guard fails or Bates already present).
        """
        if any(cit.get("bates") for cit in citations.values()):
            return 0

        match = FILENAME_BATES_RE.match(Path(source_name).name)
        if not match:
            return 0
        prefix, start_str, end_str = match.groups()
        start, end = int(start_str), int(end_str)
        width = len(start_str)

        pages = [cit.get("page") for cit in citations.values()]
        pages = [p for p in pages if p is not None]
        if not pages:
            return 0
        page_count = max(pages)
        if end - start + 1 != page_count or page_count < 1:
            return 0

        prefix = prefix.upper()
        updated = 0
        for cit in citations.values():
            page = cit.get("page")
            if page is None:
                continue
            cit["bates"] = f"{prefix}{start + page - 1:0{width}d}"
            updated += 1
        if updated:
            logger.info(
                "Derived per-page Bates %s%s-%s from filename for %d citation entries",
                prefix, start_str, end_str, updated,
            )
        return updated

    # ── Metadata Helpers ─────────────────────────────────────────────

    def _build_line_maps(
        self,
        entries: List[Tuple[str, Optional[str]]],
        citations: dict,
    ) -> Tuple[List[Optional[int]], List[Optional[str]]]:
        """
        Build per-line page_map and bates_map from chunk entries.

        Each entry is (line_text, text_id). For each line in the joined
        core_text, look up the page and bates from the citation keyed by
        text_id. Forward-fill None gaps from the last known value.

        Returns:
            (page_map, bates_map) — parallel lists, one entry per line of core_text.
        """
        page_map: List[Optional[int]] = []
        bates_map: List[Optional[str]] = []

        for _line_text, text_id in entries:
            page = None
            bates = None
            if text_id:
                cite_key = f"#/texts/{text_id}"
                cit = citations.get(cite_key, {})
                page = cit.get("page")
                bates = cit.get("bates")
            page_map.append(page)
            bates_map.append(bates)

        # Forward-fill None gaps
        last_page = None
        last_bates = None
        for i in range(len(page_map)):
            if page_map[i] is not None:
                last_page = page_map[i]
            else:
                page_map[i] = last_page
            if bates_map[i] is not None:
                last_bates = bates_map[i]
            else:
                bates_map[i] = last_bates

        return page_map, bates_map

    def _update_metadata(self, metadata: ChunkMetadata, citation: dict):
        """Update chunk metadata from a citation dict."""
        page = citation.get("page")
        if page and page not in metadata.pages:
            metadata.pages.append(page)

        bates = citation.get("bates")
        if bates and bates not in metadata.bates_stamps:
            metadata.bates_stamps.append(bates)

        line_start = citation.get("line_start")
        line_end = citation.get("line_end")
        transcript_page = citation.get("transcript_page")

        if transcript_page and line_start:
            if transcript_page not in metadata.line_ranges:
                metadata.line_ranges[transcript_page] = (line_start, line_end or line_start)
            else:
                start, end = metadata.line_ranges[transcript_page]
                metadata.line_ranges[transcript_page] = (
                    min(start, line_start),
                    max(end, line_end or line_start)
                )

        para = citation.get("paragraph_number")
        if para and para not in metadata.paragraph_numbers:
            metadata.paragraph_numbers.append(para)

        col = citation.get("column")
        if col and col not in metadata.columns:
            metadata.columns.append(col)

        tp = citation.get("transcript_page")
        if tp and tp not in metadata.transcript_pages:
            metadata.transcript_pages.append(tp)

    def _create_chunk(
        self,
        text: str,
        metadata: ChunkMetadata,
        stem: str,
        source_file: str,
        chunk_idx: int,
        doc_type: DocumentType,
        page_map: Optional[List[Optional[int]]] = None,
        bates_map: Optional[List[Optional[str]]] = None,
        source_path: str = "",
    ) -> Chunk:
        """Create a Chunk object with complete citation metadata."""
        # Generate chunk ID
        chunk_id = f"{stem}_chunk_{chunk_idx:04d}"

        # Insert explicit [PAGE:N] markers so pagination is readable inline.
        # Transcripts are excluded: they cite transcript page:line, and their
        # PDF pages (often 4-up condensed) would mislead more than help.
        if page_map and doc_type not in self.TRANSCRIPT_TYPES:
            text, page_map, bates_map = self._insert_page_markers(
                text, page_map, bates_map
            )

        # Reconcile chunk-level page/bates sets with the per-line maps, which
        # are the authoritative source when present.
        if page_map:
            map_pages = [p for p in page_map if p is not None]
            if map_pages:
                metadata.pages = sorted(set(map_pages))
        if bates_map:
            map_bates = list(dict.fromkeys(b for b in bates_map if b))
            if map_bates:
                metadata.bates_stamps = map_bates

        # Build citation dict
        citation = {
            "pdf_pages": sorted(set(metadata.pages)),
            "bates_range": metadata.bates_stamps,
        }

        # Store per-line maps for precise search attribution
        if page_map:
            citation["page_map"] = page_map
            spans = compute_page_spans(page_map)
            if spans:
                citation["page_spans"] = spans
        if bates_map and any(b is not None for b in bates_map):
            citation["bates_map"] = bates_map

        # Add type-specific fields
        if metadata.line_ranges:
            citation["transcript_lines"] = {
                str(pg): list(rng) for pg, rng in metadata.line_ranges.items()
            }
            citation["transcript_pages"] = sorted(set(metadata.transcript_pages))

        if metadata.paragraph_numbers:
            citation["paragraph_numbers"] = sorted(set(metadata.paragraph_numbers))

        if metadata.columns:
            citation["column_lines"] = {
                "columns": sorted(set(metadata.columns))
            }

        # Discovery-response fields
        if metadata.discovery_request_kind is not None:
            citation["discovery_request_kind"] = metadata.discovery_request_kind
        if metadata.discovery_request_number is not None:
            citation["discovery_request_number"] = metadata.discovery_request_number
        if metadata.discovery_part is not None:
            citation["discovery_part"] = metadata.discovery_part
        if metadata.discovery_part_total is not None:
            citation["discovery_part_total"] = metadata.discovery_part_total

        # Generate citation string
        citation_string = self._generate_citation_string(
            stem, doc_type, metadata
        )

        # Calculate tokens
        tokens = len(text) // CHARS_PER_TOKEN

        return Chunk(
            chunk_id=chunk_id,
            core_text=text,
            pages=sorted(set(metadata.pages)),
            citation=citation,
            citation_string=citation_string,
            tokens=tokens,
            doc_type=doc_type,
            source_path=source_path or None,
        )

    def _generate_citation_string(
        self,
        stem: str,
        doc_type: DocumentType,
        metadata: ChunkMetadata,
    ) -> str:
        """Generate human-readable citation string."""
        # Extract document name from stem
        doc_name = stem.replace("_", " ").title()

        # Citation format by type suffix
        TYPE_SUFFIX = {
            DocumentType.DEPOSITION: "Dep.",
            DocumentType.HEARING_TRANSCRIPT: "Tr.",
        }

        if doc_type in self.TRANSCRIPT_TYPES:
            suffix = TYPE_SUFFIX.get(doc_type, "Tr.")
            if metadata.line_ranges:
                ranges = []
                for pg in sorted(metadata.line_ranges.keys()):
                    start, end = metadata.line_ranges[pg]
                    if start == end:
                        ranges.append(f"{pg}:{start}")
                    else:
                        ranges.append(f"{pg}:{start}-{end}")
                return f"{doc_name} {suffix} {', '.join(ranges)}"
            return f"{doc_name} {suffix}"

        elif doc_type in self.PARAGRAPH_TYPES:
            if metadata.paragraph_numbers:
                paras = sorted(set(metadata.paragraph_numbers))
                if len(paras) == 1:
                    return f"{doc_name} ¶{paras[0]}"
                else:
                    return f"{doc_name} ¶¶{paras[0]}-{paras[-1]}"
            return f"{doc_name}"

        elif doc_type in self.PATENT_TYPES:
            if metadata.columns:
                cols = sorted(set(metadata.columns))
                return f"{doc_name}, col. {cols[0]}"
            return f"{doc_name}"

        elif doc_type in self.DISCOVERY_TYPES:
            kind_label = {"rog": "ROG", "rfp": "RFP", "rfa": "RFA"}.get(
                metadata.discovery_request_kind or "", "Req."
            )
            if metadata.discovery_request_number is not None:
                base = f"{doc_name} {kind_label} No. {metadata.discovery_request_number}"
                if metadata.discovery_part_total and metadata.discovery_part_total > 1:
                    base += f" (Pt. {metadata.discovery_part}/{metadata.discovery_part_total})"
                return base
            # Preamble chunk (general objections, definitions) before any
            # numbered request.
            if metadata.pages:
                return f"{doc_name} (preamble), {format_page_ranges(metadata.pages)}"
            return f"{doc_name} (preamble)"

        else:
            # Generic: Prefer Bates numbers over page numbers (legal convention - Bug Fix #2)
            if metadata.bates_stamps:
                # Use Bates number format: "Document at BATES_001" or "Document at BATES_001-BATES_003"
                bates_list = sorted(set(metadata.bates_stamps))
                if len(bates_list) == 1:
                    return f"{doc_name} at {bates_list[0]}"
                else:
                    return f"{doc_name} at {bates_list[0]}-{bates_list[-1]}"

            # Fallback to page numbers if no Bates stamps. Preserve gaps
            # ("pp. 1, 6, 8-10") rather than collapsing to a min-max span.
            if metadata.pages:
                return f"{doc_name}, {format_page_ranges(metadata.pages)}"

            return f"{doc_name}"


def chunk_all_documents(
    converted_dir: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    doc_type_map: Optional[Dict[str, DocumentType]] = None,
    source_path_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Chunk]]:
    """
    Chunk all documents in a converted directory.

    Args:
        converted_dir: Directory containing .md and _citations.json files
        target_tokens: Target chunk size in tokens
        doc_type_map: Pre-computed mapping of stem -> DocumentType from classifier
        source_path_map: Pre-computed mapping of stem -> original relative path

    Returns:
        Dict mapping stem to list of chunks
    """
    converted_dir = Path(converted_dir)
    chunker = DocumentChunker(str(converted_dir), target_tokens=target_tokens)
    if doc_type_map is None:
        doc_type_map = {}
    if source_path_map is None:
        source_path_map = {}

    results = {}
    for md_file in sorted(converted_dir.glob("*.md")):
        stem = md_file.stem

        # Skip if citations file doesn't exist
        citations_file = converted_dir / f"{stem}_citations.json"
        if not citations_file.exists():
            logger.warning("No citations file for %s, skipping", stem)
            continue

        # Look up doc type from classifier map, fall back to inferring from citations
        doc_type = doc_type_map.get(stem, DocumentType.UNKNOWN)
        if doc_type == DocumentType.UNKNOWN:
            doc_type = _infer_type_from_citations(citations_file)

        source_path = source_path_map.get(stem, "")
        chunks = chunker.chunk_document(stem, doc_type, md_file.name, source_path=source_path)
        results[stem] = chunks

    return results


def _infer_type_from_citations(citations_path: Path) -> DocumentType:
    """Infer document type from citation type fields in _citations.json."""
    try:
        with open(citations_path) as f:
            citations = json.load(f)
    except (json.JSONDecodeError, OSError):
        return DocumentType.UNKNOWN

    type_counts: Dict[str, int] = {}
    for cit in citations.values():
        t = cit.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    if "transcript_line" in type_counts:
        return DocumentType.DEPOSITION
    if "patent_column" in type_counts:
        return DocumentType.PATENT
    if "discovery_request" in type_counts:
        return DocumentType.DISCOVERY_RESPONSE
    if "paragraph" in type_counts and type_counts["paragraph"] > 3:
        return DocumentType.EXPERT_REPORT
    return DocumentType.UNKNOWN
