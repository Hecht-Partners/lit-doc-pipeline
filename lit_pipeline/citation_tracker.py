"""
Citation reconstruction from Docling JSON output.

Reconstructs line-level (depositions), column-level (patents), and paragraph-level
(expert reports) citation metadata using bbox geometry from Docling's JSON provenance data.

CRITICAL: Docling strips line numbers from depositions and merges text blobs.
This module uses bbox-based line inference as the PRIMARY method, not regex fallback.

Runs AFTER post-processing, BEFORE chunking. Outputs {stem}_citations.json.
"""

import json
import logging
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lit_pipeline.citation_types import CitationData, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class PageLayout:
    """Per-page geometric parameters computed from well-populated pages."""
    content_top: float = 655.0
    content_bottom: float = 87.5
    lines_per_page: int = 25

    @property
    def line_height(self) -> float:
        if self.lines_per_page <= 0:
            return 1.0
        return (self.content_top - self.content_bottom) / self.lines_per_page

    def bbox_to_lines(self, bbox_t: float, bbox_b: float) -> Tuple[int, int]:
        """Convert bbox top/bottom to line_start/line_end (1-indexed, clamped)."""
        lh = self.line_height
        if lh <= 0:
            return (1, 1)
        line_start = math.floor((self.content_top - bbox_t) / lh) + 1
        line_end = math.floor((self.content_top - bbox_b) / lh)
        line_start = max(1, min(line_start, self.lines_per_page))
        line_end = max(line_start, min(line_end, self.lines_per_page))
        return (line_start, line_end)


@dataclass
class ValidationMetrics:
    """Metrics from citation validation."""
    total_items: int = 0
    coverage_pct: float = 0.0
    has_line_numbers: bool = False
    has_column_numbers: bool = False
    has_bates: bool = False
    type_distribution: Dict[str, int] = field(default_factory=dict)
    line_gaps: List[str] = field(default_factory=list)
    bates_gaps: List[str] = field(default_factory=list)
    bates_duplicates: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "coverage_pct": self.coverage_pct,
            "has_line_numbers": self.has_line_numbers,
            "has_column_numbers": self.has_column_numbers,
            "has_bates": self.has_bates,
            "type_distribution": self.type_distribution,
            "line_gaps": self.line_gaps,
            "bates_gaps": self.bates_gaps,
            "bates_duplicates": self.bates_duplicates,
        }


class CitationTracker:
    """
    Reconstructs citation metadata from Docling JSON provenance data.

    Uses bbox geometry to infer line numbers (depositions), column positions
    (patents), and paragraph numbers (expert reports).
    """

    def __init__(
        self,
        converted_dir: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        lines_per_page: int = 25,
    ):
        self.converted_dir = Path(converted_dir)
        self.doc_type = doc_type
        self.lines_per_page = lines_per_page

    def reconstruct_citations(self, stem: str) -> Dict[str, dict]:
        """
        Main entry point. Load inputs, route to handler, save output.

        Args:
            stem: File stem (e.g. "daniel_alexander_10_24_2025")

        Returns:
            Dict mapping self_ref strings to citation dicts.
        """
        doc_data, bates_by_page = self._load_inputs(stem)
        if doc_data is None:
            return {}

        texts = doc_data.get("texts", [])
        if not texts:
            logger.warning("No texts array in JSON for %s", stem)
            return {}

        self_ref_map = self._build_self_ref_map(texts)

        TRANSCRIPT_TYPES = {DocumentType.DEPOSITION, DocumentType.HEARING_TRANSCRIPT}
        PARAGRAPH_TYPES = {
            DocumentType.EXPERT_REPORT, DocumentType.PLEADING,
            DocumentType.DECLARATION, DocumentType.MOTION,
            DocumentType.BRIEF, DocumentType.WITNESS_STATEMENT,
            DocumentType.AGREEMENT,
        }
        PATENT_TYPES = {DocumentType.PATENT}
        DISCOVERY_TYPES = {DocumentType.DISCOVERY_RESPONSE, DocumentType.DISCOVERY_REQUEST}

        if self.doc_type in TRANSCRIPT_TYPES:
            citations = self._handle_deposition(texts, self_ref_map, bates_by_page)
        elif self.doc_type in PATENT_TYPES:
            citations = self._handle_patent(texts, self_ref_map, bates_by_page, doc_data)
        elif self.doc_type in DISCOVERY_TYPES:
            citations = self._handle_discovery(texts, self_ref_map, bates_by_page)
        elif self.doc_type in PARAGRAPH_TYPES:
            citations = self._handle_expert_report(texts, self_ref_map, bates_by_page)
        else:
            citations = self._handle_generic(texts, self_ref_map, bates_by_page)

        # Save output
        out_path = self.converted_dir / f"{stem}_citations.json"
        with open(out_path, "w") as f:
            json.dump(citations, f, indent=2)
        logger.info("Saved %d citations to %s", len(citations), out_path)

        # Validate and log metrics
        metrics = self.validate(citations)
        logger.info(
            "Citation metrics: %d items, %.1f%% coverage, types=%s",
            metrics.total_items,
            metrics.coverage_pct,
            metrics.type_distribution,
        )
        if metrics.line_gaps:
            logger.warning("Line gaps detected: %s", metrics.line_gaps[:5])
        if metrics.bates_gaps:
            logger.warning("Bates stamp gaps detected: %s", metrics.bates_gaps[:5])
        if metrics.bates_duplicates:
            logger.warning("Bates stamp duplicates detected: %s", metrics.bates_duplicates[:5])

        return citations

    # ── Loading ──────────────────────────────────────────────────────────

    def _load_inputs(self, stem: str) -> Tuple[Optional[dict], Dict[str, str]]:
        """Load Docling JSON and extract Bates stamps from page footers."""
        json_path = self.converted_dir / f"{stem}.json"
        if not json_path.exists():
            logger.warning("No JSON file found at %s", json_path)
            return None, {}

        with open(json_path) as f:
            doc_data = json.load(f)

        # Extract Bates stamps from page_footer elements in JSON
        bates_by_page = self._extract_bates_from_json(doc_data)

        # Also check for optional Bates sidecar (legacy support)
        bates_path = self.converted_dir / f"{stem}_bates.json"
        if bates_path.exists():
            with open(bates_path) as f:
                sidecar_bates = json.load(f)
                # Merge sidecar into extracted (sidecar takes precedence)
                bates_by_page.update(sidecar_bates)

        if bates_by_page:
            logger.info("Extracted Bates stamps for %d pages", len(bates_by_page))

        return doc_data, bates_by_page

    def _extract_bates_from_json(self, doc_data: dict) -> Dict[str, str]:
        """Extract Bates stamps from page_footer elements in Docling JSON.

        Returns:
            Dict mapping page_no (as string) to Bates stamp
        """
        bates_patterns = [
            re.compile(r'INTEL_PROX_\d{5,11}'),
            re.compile(r'PROX_INTEL[-_]\d{5,11}'),
            re.compile(r'[A-Z]{2,}[-_][A-Z]{2,}[-_]\d{5,}')
        ]

        bates_by_page: Dict[str, str] = {}
        texts = doc_data.get("texts", [])

        for elem in texts:
            # Look for page_footer elements
            if elem.get("label") != "page_footer":
                continue

            text = elem.get("text", "").strip()
            page_no = self._get_page_no(elem)

            if not text or page_no is None:
                continue

            # Try to match Bates pattern
            for pattern in bates_patterns:
                match = pattern.search(text)
                if match:
                    bates_stamp = match.group(0)
                    bates_by_page[str(page_no)] = bates_stamp
                    break

        return bates_by_page

    def _build_self_ref_map(self, texts: list) -> Dict[int, str]:
        """Map array index to self_ref string.

        CRITICAL: texts[0] may have self_ref="#/texts/25", NOT "#/texts/0".
        We must use self_ref as keys in the output, not array indices.
        """
        result = {}
        for i, t in enumerate(texts):
            self_ref = t.get("self_ref", f"#/texts/{i}")
            result[i] = self_ref
        return result

    def _associate_bates(self, page_no: int, bates_by_page: Dict[str, str]) -> Optional[str]:
        """Look up Bates stamp for a given page number."""
        return bates_by_page.get(str(page_no))

    # ── Element Classification ───────────────────────────────────────────

    def _get_bbox(self, elem: dict) -> Optional[dict]:
        """Extract bbox from element's first provenance entry."""
        prov = elem.get("prov", [])
        if not prov:
            return None
        return prov[0].get("bbox")

    def _get_page_no(self, elem: dict) -> Optional[int]:
        """Extract page number from element's first provenance entry."""
        prov = elem.get("prov", [])
        if not prov:
            return None
        return prov[0].get("page_no")

    def _is_page_marker(self, elem: dict) -> Optional[int]:
        """Detect "Page N" markers by text match AND bbox position.

        Returns transcript page number if it's a marker, None otherwise.
        Uses bbox position (l > 400 or t > 700) to avoid false positives
        from conversational "Page 18" references in Q/A text.
        """
        text = elem.get("text", "").strip()
        match = re.match(r"^Page\s+(\d+)$", text)
        if not match:
            return None

        bbox = self._get_bbox(elem)
        if not bbox:
            return int(match.group(1))

        # Page markers are typically in the upper-right: l > 400 or t > 700
        if bbox.get("l", 0) > 400 or bbox.get("t", 0) > 700:
            return int(match.group(1))

        return None

    def _is_header_or_marker(self, elem: dict) -> bool:
        """Detect headers (CONFIDENTIAL...) and name headers (section_header label)."""
        label = elem.get("label", "")
        if label == "section_header":
            return True

        text = elem.get("text", "").strip().upper()
        if text.startswith("CONFIDENTIAL"):
            return True

        return False

    def _is_skippable(self, elem: dict) -> bool:
        """Check if element should be skipped entirely."""
        text = elem.get("text", "").strip()
        if not text:
            return True

        bbox = self._get_bbox(elem)
        if bbox:
            height = abs(bbox.get("t", 0) - bbox.get("b", 0))
            if height < 3:
                return True

        return False

    def _detect_page_markers(self, texts: list) -> Dict[int, int]:
        """Walk texts array, return {array_idx: transcript_page_number}."""
        markers = {}
        for i, t in enumerate(texts):
            page_num = self._is_page_marker(t)
            if page_num is not None:
                markers[i] = page_num
        return markers

    # ── Deposition Handler ───────────────────────────────────────────────

    def _compute_content_area(self, texts: list) -> PageLayout:
        """Compute standard content area from well-populated pages.

        Uses pages with >= 8 content elements (excluding headers/markers)
        to get robust median values for content_top and content_bottom.
        """
        page_tops: Dict[int, List[float]] = defaultdict(list)
        page_bottoms: Dict[int, List[float]] = defaultdict(list)

        for t in texts:
            if self._is_header_or_marker(t):
                continue
            if self._is_page_marker(t) is not None:
                continue
            if self._is_skippable(t):
                continue

            bbox = self._get_bbox(t)
            page_no = self._get_page_no(t)
            if not bbox or page_no is None:
                continue

            page_tops[page_no].append(bbox.get("t", 0))
            page_bottoms[page_no].append(bbox.get("b", 0))

        # Use pages with >= 8 content elements for robust statistics
        good_tops = []
        good_bottoms = []
        for pg in page_tops:
            if len(page_tops[pg]) >= 8:
                good_tops.append(max(page_tops[pg]))
                good_bottoms.append(min(page_bottoms[pg]))

        if not good_tops:
            logger.warning("No well-populated pages found, using defaults")
            return PageLayout(lines_per_page=self.lines_per_page)

        content_top = statistics.median(good_tops)
        content_bottom = statistics.median(good_bottoms)

        layout = PageLayout(
            content_top=content_top,
            content_bottom=content_bottom,
            lines_per_page=self.lines_per_page,
        )
        logger.info(
            "Content area: top=%.1f bottom=%.1f height=%.1f line_height=%.1f",
            layout.content_top,
            layout.content_bottom,
            layout.content_top - layout.content_bottom,
            layout.line_height,
        )
        return layout

    def _parse_code_block_lines(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract line range from code blocks with embedded line numbers.

        Handles the rare case where Docling merges deposition lines into
        a single 'code' element like:
            "1 DANIEL ALEXANDER 2 Do you know what..."
        """
        matches = re.findall(r"(?:^|\s)(\d{1,2})\s+[A-Za-z]", text)
        if not matches:
            return None

        line_nums = []
        for m in matches:
            n = int(m)
            if 1 <= n <= 25:
                line_nums.append(n)

        if not line_nums:
            return None

        return (min(line_nums), max(line_nums))

    def _handle_deposition(
        self,
        texts: list,
        self_ref_map: Dict[int, str],
        bates_by_page: Dict[str, str],
    ) -> Dict[str, dict]:
        """Process deposition using bbox-based line inference."""
        layout = self._compute_content_area(texts)
        page_markers = self._detect_page_markers(texts)

        citations: Dict[str, dict] = {}
        current_transcript_page: Optional[int] = None

        for i, t in enumerate(texts):
            # Update transcript page from page markers
            if i in page_markers:
                current_transcript_page = page_markers[i]
                continue

            if self._is_header_or_marker(t):
                continue
            if self._is_skippable(t):
                continue

            bbox = self._get_bbox(t)
            page_no = self._get_page_no(t)
            if not bbox or page_no is None:
                continue

            self_ref = self_ref_map.get(i, f"#/texts/{i}")

            # Use transcript page if available, else fall back to PDF page
            transcript_page = current_transcript_page if current_transcript_page else page_no

            # Handle code blocks with embedded line numbers
            label = t.get("label", "")
            if label == "code":
                code_lines = self._parse_code_block_lines(t.get("text", ""))
                if code_lines:
                    citations[self_ref] = CitationData(
                        page=page_no,
                        line_start=code_lines[0],
                        line_end=code_lines[1],
                        bates=self._associate_bates(page_no, bates_by_page),
                        type="transcript_line",
                    ).to_dict()
                    citations[self_ref]["transcript_page"] = transcript_page
                    continue

            # Primary: bbox-based line inference
            bbox_t = bbox.get("t", 0)
            bbox_b = bbox.get("b", 0)
            line_start, line_end = layout.bbox_to_lines(bbox_t, bbox_b)

            citations[self_ref] = CitationData(
                page=page_no,
                line_start=line_start,
                line_end=line_end,
                bates=self._associate_bates(page_no, bates_by_page),
                type="transcript_line",
            ).to_dict()
            citations[self_ref]["transcript_page"] = transcript_page

        return citations

    # ── Patent Handler ───────────────────────────────────────────────────

    def _detect_patent_columns(
        self, texts: list, doc_data: dict
    ) -> Tuple[float, float]:
        """Detect page width and content midpoint from patent data.

        Returns (page_width, content_midpoint) for column classification.
        """
        pages = doc_data.get("pages", {})
        if pages:
            first_page = pages.get("1") or pages.get(list(pages.keys())[0])
            page_width = first_page.get("size", {}).get("width", 612.0)
        else:
            page_width = 612.0

        # Compute content midpoint from actual element positions
        left_positions = []
        right_positions = []
        for t in texts:
            bbox = self._get_bbox(t)
            if not bbox:
                continue
            text = t.get("text", "")
            if len(text) < 20:
                continue
            l = bbox.get("l", 0)
            r = bbox.get("r", 0)
            width = r - l
            # Skip very wide elements (full-width)
            if width > page_width * 0.55:
                continue
            center = (l + r) / 2
            if center < page_width * 0.4:
                left_positions.append(r)
            elif center > page_width * 0.6:
                right_positions.append(l)

        if left_positions and right_positions:
            content_midpoint = (statistics.median(left_positions) + statistics.median(right_positions)) / 2
        else:
            content_midpoint = page_width * 0.5

        return page_width, content_midpoint

    def _classify_patent_element(
        self, bbox: dict, page_width: float, content_midpoint: float
    ) -> Optional[str]:
        """Classify element as 'left', 'right', 'full_width', or None (ambiguous)."""
        l = bbox.get("l", 0)
        r = bbox.get("r", 0)
        width = r - l

        if width > page_width * 0.65:
            return "full_width"

        center = (l + r) / 2
        if r < content_midpoint and l < page_width * 0.38:
            return "left"
        if l > content_midpoint * 0.85 and l >= page_width * 0.38:
            return "right"

        # Ambiguous: use center vs midpoint
        if center < content_midpoint:
            return "left"
        return "right"

    def _handle_patent(
        self,
        texts: list,
        self_ref_map: Dict[int, str],
        bates_by_page: Dict[str, str],
        doc_data: dict,
    ) -> Dict[str, dict]:
        """Process patent with column detection."""
        page_width, content_midpoint = self._detect_patent_columns(texts, doc_data)
        logger.info(
            "Patent layout: page_width=%.1f, content_midpoint=%.1f",
            page_width,
            content_midpoint,
        )

        # Classify pages as figure/spec/claims based on text density
        page_text_length: Dict[int, int] = defaultdict(int)
        page_elements_count: Dict[int, int] = defaultdict(int)
        for t in texts:
            page_no = self._get_page_no(t)
            if page_no is None:
                continue
            page_text_length[page_no] += len(t.get("text", ""))
            page_elements_count[page_no] += 1

        # Spec pages have substantial text; figure pages have many short labels
        spec_pages = set()
        for pg, length in page_text_length.items():
            elem_count = page_elements_count.get(pg, 0)
            avg_len = length / elem_count if elem_count > 0 else 0
            if length > 300 and avg_len > 30:
                spec_pages.add(pg)

        # Determine first spec page for column numbering
        first_spec_page = min(spec_pages) if spec_pages else 1

        # Compute per-column line layout (typically ~60-65 lines per column)
        col_layout = PageLayout(
            content_top=700.0,
            content_bottom=80.0,
            lines_per_page=65,
        )
        # Refine from actual data if possible
        col_tops: List[float] = []
        col_bottoms: List[float] = []
        for t in texts:
            page_no = self._get_page_no(t)
            if page_no not in spec_pages:
                continue
            bbox = self._get_bbox(t)
            if not bbox:
                continue
            col_tops.append(bbox.get("t", 0))
            col_bottoms.append(bbox.get("b", 0))

        if col_tops:
            col_layout.content_top = max(col_tops)
            col_layout.content_bottom = min(col_bottoms)

        citations: Dict[str, dict] = {}

        for i, t in enumerate(texts):
            if self._is_skippable(t):
                continue

            bbox = self._get_bbox(t)
            page_no = self._get_page_no(t)
            if not bbox or page_no is None:
                continue

            self_ref = self_ref_map.get(i, f"#/texts/{i}")
            bates = self._associate_bates(page_no, bates_by_page)

            # Figure pages: just page + Bates
            if page_no not in spec_pages:
                citations[self_ref] = CitationData(
                    page=page_no,
                    bates=bates,
                    type="patent_figure",
                ).to_dict()
                continue

            # Classify column position
            col_pos = self._classify_patent_element(bbox, page_width, content_midpoint)

            if col_pos == "full_width":
                citations[self_ref] = CitationData(
                    page=page_no,
                    bates=bates,
                    type="cross_column_merge",
                ).to_dict()
                continue

            # Compute column number: sequential across pages
            page_offset = page_no - first_spec_page
            if col_pos == "left":
                column = page_offset * 2 + 1
            else:
                column = page_offset * 2 + 2

            # Compute line within column
            bbox_t = bbox.get("t", 0)
            bbox_b = bbox.get("b", 0)
            line_start, line_end = col_layout.bbox_to_lines(bbox_t, bbox_b)

            citations[self_ref] = CitationData(
                page=page_no,
                column=column,
                line_start=line_start,
                line_end=line_end,
                bates=bates,
                type="patent_column",
            ).to_dict()

        return citations

    # ── Expert Report Handler ────────────────────────────────────────────

    def _handle_expert_report(
        self,
        texts: list,
        self_ref_map: Dict[int, str],
        bates_by_page: Dict[str, str],
    ) -> Dict[str, dict]:
        """Process expert reports / pleadings with paragraph tracking."""
        citations: Dict[str, dict] = {}
        current_paragraph: Optional[int] = None

        # Patterns for paragraph markers (in priority order):
        # 1. ¶ N or § N (symbol-based)
        # 2. "Paragraph N" (word-based)
        # 3. "N. " at start of text (numbered paragraphs)
        symbol_pattern = re.compile(r"(?:^|\s)([¶§])\s*(\d+)")
        word_pattern = re.compile(r"(?:^|\s)[Pp]aragraph\s+(\d+)")
        numbered_pattern = re.compile(r"^(\d+)\.\s+[A-Z]")  # "1. " at start with capital letter

        for i, t in enumerate(texts):
            if self._is_skippable(t):
                continue

            page_no = self._get_page_no(t)
            if page_no is None:
                continue

            self_ref = self_ref_map.get(i, f"#/texts/{i}")
            text = t.get("text", "")

            # Check for paragraph markers (priority order)
            para_num = None

            # Try symbol pattern first (¶ N, § N)
            match = symbol_pattern.search(text)
            if match:
                para_num = int(match.group(2))

            # Try word pattern (Paragraph N)
            if para_num is None:
                match = word_pattern.search(text)
                if match:
                    para_num = int(match.group(1))

            # Try numbered paragraph pattern (N. )
            # Only update if it's a reasonable increment (not a list or subsection)
            if para_num is None:
                match = numbered_pattern.match(text)
                if match:
                    candidate = int(match.group(1))
                    # Accept if: first paragraph OR reasonable increment from previous
                    if current_paragraph is None or (1 <= candidate <= 200 and candidate >= current_paragraph):
                        para_num = candidate

            if para_num is not None:
                current_paragraph = para_num

            bates = self._associate_bates(page_no, bates_by_page)

            citations[self_ref] = CitationData(
                page=page_no,
                paragraph_number=current_paragraph,
                bates=bates,
                type="paragraph" if current_paragraph else "page_only",
            ).to_dict()

        return citations

    # ── Discovery Handler ────────────────────────────────────────────────

    # Boundary markers in discovery requests/responses. Group(1) captures the
    # request kind so we can store it as kind=rog/rfp/rfa.
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
    def _kind_from_label(label: str) -> str:
        u = label.upper()
        if "PRODUCTION" in u:
            return "rfp"
        if "ADMISSION" in u:
            return "rfa"
        return "rog"

    def _handle_discovery(
        self,
        texts: list,
        self_ref_map: Dict[int, str],
        bates_by_page: Dict[str, str],
    ) -> Dict[str, dict]:
        """Discovery requests / responses. Tracks numbered request boundaries.

        Each text element inherits the most recently seen request number,
        so chunkers can group all text under one INTERROGATORY/RFP/RFA NO. N
        into a single chunk.
        """
        citations: Dict[str, dict] = {}
        current_kind: Optional[str] = None
        current_number: Optional[int] = None

        for i, t in enumerate(texts):
            if self._is_skippable(t):
                continue

            page_no = self._get_page_no(t)
            if page_no is None:
                continue

            self_ref = self_ref_map.get(i, f"#/texts/{i}")
            text = t.get("text", "")

            match = self.DISCOVERY_BOUNDARY_RE.match(text)
            if match:
                current_kind = self._kind_from_label(match.group(1))
                current_number = int(match.group(2))

            bates = self._associate_bates(page_no, bates_by_page)

            citations[self_ref] = CitationData(
                page=page_no,
                bates=bates,
                type="discovery_request" if current_number is not None else "page_only",
                discovery_request_kind=current_kind,
                discovery_request_number=current_number,
            ).to_dict()

        return citations

    # ── Generic Handler ──────────────────────────────────────────────────

    def _handle_generic(
        self,
        texts: list,
        self_ref_map: Dict[int, str],
        bates_by_page: Dict[str, str],
    ) -> Dict[str, dict]:
        """Fallback: page + Bates only."""
        citations: Dict[str, dict] = {}

        for i, t in enumerate(texts):
            if self._is_skippable(t):
                continue

            page_no = self._get_page_no(t)
            if page_no is None:
                continue

            self_ref = self_ref_map.get(i, f"#/texts/{i}")

            citations[self_ref] = CitationData(
                page=page_no,
                bates=self._associate_bates(page_no, bates_by_page),
                type="page_only",
            ).to_dict()

        return citations

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self, citations: Dict[str, dict]) -> ValidationMetrics:
        """Validate citations and return metrics."""
        if not citations:
            return ValidationMetrics()

        total = len(citations)
        type_counter = Counter()
        has_lines = False
        has_columns = False
        has_bates = False

        # Track line continuity per transcript page (depositions)
        page_lines: Dict[int, List[int]] = defaultdict(list)

        for ref, cit in citations.items():
            type_counter[cit.get("type", "unknown")] += 1

            if cit.get("line_start") is not None:
                has_lines = True
            if cit.get("column") is not None:
                has_columns = True
            if cit.get("bates"):
                has_bates = True

            tp = cit.get("transcript_page")
            ls = cit.get("line_start")
            if tp is not None and ls is not None:
                page_lines[tp].append(ls)

        # Check for line gaps in depositions
        line_gaps = []
        for pg in sorted(page_lines.keys()):
            lines = sorted(set(page_lines[pg]))
            if len(lines) >= 2:
                for j in range(len(lines) - 1):
                    gap = lines[j + 1] - lines[j]
                    if gap > 5:
                        line_gaps.append(f"Page {pg}: gap {lines[j]}->{lines[j+1]}")

        # Check for Bates stamp sequential validation
        bates_gaps = []
        bates_duplicates = []
        if has_bates:
            bates_by_page: Dict[int, str] = {}
            for cit in citations.values():
                page = cit.get("page")
                bates = cit.get("bates")
                if page and bates:
                    if page in bates_by_page:
                        if bates_by_page[page] != bates:
                            bates_duplicates.append(f"Page {page}: multiple Bates stamps")
                    else:
                        bates_by_page[page] = bates

            # Extract numeric suffix from Bates stamps and check for gaps
            bates_numbers = {}
            for page, bates in sorted(bates_by_page.items()):
                # Try to extract numeric suffix
                match = re.search(r'(\d{5,})$', bates)
                if match:
                    bates_numbers[page] = int(match.group(1))

            # Check for sequential gaps
            if len(bates_numbers) >= 2:
                pages = sorted(bates_numbers.keys())
                for j in range(len(pages) - 1):
                    p1, p2 = pages[j], pages[j + 1]
                    b1, b2 = bates_numbers[p1], bates_numbers[p2]
                    expected_gap = p2 - p1
                    actual_gap = b2 - b1
                    # Allow for single-page increments (b2 = b1 + 1)
                    if actual_gap > expected_gap + 5:
                        bates_gaps.append(
                            f"Pages {p1}-{p2}: Bates {b1}->{b2} (gap: {actual_gap - expected_gap})"
                        )

        # Coverage = % of items with at least line_start or paragraph_number or column
        items_with_detail = sum(
            1
            for cit in citations.values()
            if cit.get("line_start") is not None
            or cit.get("paragraph_number") is not None
            or cit.get("column") is not None
        )
        coverage_pct = (items_with_detail / total * 100) if total > 0 else 0.0

        return ValidationMetrics(
            total_items=total,
            coverage_pct=coverage_pct,
            has_line_numbers=has_lines,
            has_column_numbers=has_columns,
            has_bates=has_bates,
            type_distribution=dict(type_counter),
            line_gaps=line_gaps,
            bates_gaps=bates_gaps,
            bates_duplicates=bates_duplicates,
        )


def reconstruct_citations(
    converted_dir: str,
    stem: str,
    doc_type: DocumentType = DocumentType.UNKNOWN,
    lines_per_page: int = 25,
) -> Dict[str, dict]:
    """Convenience function: create tracker and reconstruct citations."""
    tracker = CitationTracker(
        converted_dir=converted_dir,
        doc_type=doc_type,
        lines_per_page=lines_per_page,
    )
    return tracker.reconstruct_citations(stem)
