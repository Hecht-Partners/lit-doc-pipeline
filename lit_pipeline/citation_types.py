"""
Citation data structures and types for the litigation document processing pipeline.

This module defines the core data structures for tracking citations throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


class DocumentType(Enum):
    """Supported document types with different citation formats."""
    DEPOSITION = "deposition"           # page:line format (e.g., "45:12-18")
    PATENT = "patent"                   # column:line format (e.g., "col. 3:45-55")
    EXPERT_REPORT = "expert_report"     # paragraph format (e.g., "¶ 42")
    PLEADING = "pleading"               # paragraph format (e.g., "Complaint ¶ 23")
    EXHIBIT = "exhibit"                 # page format (e.g., "Ex. 15, p. 3")
    UNKNOWN = "unknown"
    # Extended types
    HEARING_TRANSCRIPT = "hearing_transcript"       # page:line (reuses deposition handler)
    DECLARATION = "declaration"                     # ¶ N (reuses expert_report handler)
    MOTION = "motion"                               # ¶ N or p. N
    BRIEF = "brief"                                 # ¶ N or p. N
    PATENT_PROSECUTION = "patent_prosecution"       # p. N (generic handler)
    AGREEMENT = "agreement"                         # § N.N, p. N
    CORRESPONDENCE = "correspondence"               # p. N (generic handler)
    TECHNICAL_STANDARD = "technical_standard"       # § N.N, p. N (generic handler)
    SCHOLARLY = "scholarly"                          # p. N (generic handler)
    SEC_FILING = "sec_filing"                       # p. N (generic handler)
    WITNESS_STATEMENT = "witness_statement"          # ¶ N (reuses expert_report handler)
    DISCOVERY_RESPONSE = "discovery_response"        # ROG/RFP/RFA No. N format
    DISCOVERY_REQUEST = "discovery_request"          # ROG/RFP/RFA No. N (no responses)


@dataclass
class CitationData:
    """
    Citation metadata for a text element.

    Different document types populate different fields:
    - Depositions: page, line_start, line_end, bates
    - Patents: page, column, line_start, line_end
    - Expert Reports: page, paragraph_number
    """
    page: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    bates: Optional[str] = None
    column: Optional[int] = None
    paragraph_number: Optional[int] = None
    type: str = "unknown"
    # Discovery-response fields: identifies which numbered request a text element
    # belongs to. kind is one of {"rog", "rfp", "rfa"}; number is 1, 2, ...
    discovery_request_kind: Optional[str] = None
    discovery_request_number: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "page": self.page,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "bates": self.bates,
            "column": self.column,
            "paragraph_number": self.paragraph_number,
            "type": self.type
        }
        if self.discovery_request_kind is not None:
            d["discovery_request_kind"] = self.discovery_request_kind
        if self.discovery_request_number is not None:
            d["discovery_request_number"] = self.discovery_request_number
        return d


@dataclass
class ConversionResult:
    """Result of document conversion with citation extraction."""
    md_path: str
    citations_found: Dict[str, List]
    needs_reconstruction: bool
    doc_type: DocumentType = DocumentType.UNKNOWN
    errors: List[str] = field(default_factory=list)

    def citation_coverage_summary(self) -> str:
        """Generate a summary of citation coverage."""
        lines = ["Citation Coverage Summary:"]
        lines.append(f"  Document Type: {self.doc_type.value}")
        lines.append(f"  Pages Found: {len(self.citations_found.get('pages', []))}")
        lines.append(f"  Line Markers: {len(self.citations_found.get('line_markers', []))}")
        lines.append(f"  Bates Stamps: {len(self.citations_found.get('bates_stamps', []))}")
        lines.append(f"  Column Markers: {len(self.citations_found.get('column_markers', []))}")
        lines.append(f"  Paragraph Markers: {len(self.citations_found.get('paragraph_markers', []))}")
        lines.append(f"  Needs Reconstruction: {'Yes' if self.needs_reconstruction else 'No'}")

        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for error in self.errors[:3]:  # Show first 3 errors
                lines.append(f"    - {error}")

        return "\n".join(lines)


@dataclass
class ProcessingResult:
    """Result of post-processing step."""
    cleaned_path: str
    citations_path: str
    citation_coverage: int
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Processing Result:",
            f"  Cleaned File: {self.cleaned_path}",
            f"  Citations File: {self.citations_path}",
            f"  Citation Coverage: {self.citation_coverage} elements"
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:3]:
                lines.append(f"    - {warning}")
        return "\n".join(lines)


@dataclass
class Chunk:
    """
    A semantic chunk of text with citation metadata and deterministic quotes.

    CRITICAL: key_quotes are extracted deterministically from core_text,
    NOT generated by LLM (to avoid hallucination).

    Enrichment fields (set by LLM enrichment, optional):
    - summary: 2-3 sentence summary of legal significance
    - category: one of the VALID_CATEGORIES
    - relevance_score: high/medium/low
    - claims_addressed: list of patent claim numbers
    - classification_method: "llm" or "keyword"
    - llm_backend: "ollama", "anthropic", or "claude_code"
    """
    chunk_id: str
    core_text: str
    pages: List[int]
    citation: Dict[str, any]
    citation_string: str
    key_quotes: List[str] = field(default_factory=list)
    tokens: int = 0
    doc_type: DocumentType = DocumentType.UNKNOWN
    # Enrichment fields (optional, set by LLM enrichment)
    summary: Optional[str] = None
    category: Optional[str] = None
    relevance_score: Optional[str] = None
    claims_addressed: Optional[List[int]] = None
    classification_method: Optional[str] = None
    llm_backend: Optional[str] = None
    source_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "chunk_id": self.chunk_id,
            "core_text": self.core_text,
            "pages": self.pages,
            "citation": self.citation,
            "citation_string": self.citation_string,
            "key_quotes": self.key_quotes,
            "tokens": self.tokens,
            "doc_type": self.doc_type.value
        }
        # Include enrichment fields only when set (keeps unenriched chunks clean)
        if self.summary is not None:
            d["summary"] = self.summary
        if self.category is not None:
            d["category"] = self.category
        if self.relevance_score is not None:
            d["relevance_score"] = self.relevance_score
        if self.claims_addressed is not None:
            d["claims_addressed"] = self.claims_addressed
        if self.classification_method is not None:
            d["classification_method"] = self.classification_method
        if self.llm_backend is not None:
            d["llm_backend"] = self.llm_backend
        if self.source_path is not None:
            d["source_path"] = self.source_path
        return d


@dataclass
class SearchResult:
    """
    A search result with score and ranking information.

    Used by hybrid retrieval to return chunks with relevance scores
    from BM25, semantic search, or both.
    """
    chunk_id: str
    chunk: Chunk
    score: float
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    reranker_score: Optional[float] = None
    rank: int = 0
