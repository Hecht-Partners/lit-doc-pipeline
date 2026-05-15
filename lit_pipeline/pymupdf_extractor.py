"""
PyMuPDF-based deposition extractor for text-based PDFs.

Provides exact line-by-line extraction with perfect line number mapping,
bypassing Docling's bbox-based line inference which loses granularity
when multi-line text gets merged into blobs.

Only used for text-based depositions (e.g., Alexander). Scanned PDFs
still require Docling+OCR.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from lit_pipeline.citation_types import CitationData

logger = logging.getLogger(__name__)

# Thresholds for span classification
MARGIN_X_THRESHOLD = 100  # Line numbers appear at x0 < 100
CONTENT_X_THRESHOLD = 100  # Content text starts at x0 >= 100
Y_GROUP_TOLERANCE = 3  # px tolerance for grouping spans into visual lines
PAGE_MARKER_X_THRESHOLD = 400  # "Page N" markers appear at x0 > 400
PAGE_MARKER_Y_THRESHOLD = 95  # "Page N" near top of page (y < 95)
HEADER_Y_THRESHOLD = 70  # CONFIDENTIAL headers near very top
FOOTER_Y_THRESHOLD = 730  # Footer area (phone numbers, URLs)


def is_text_based_pdf(pdf_path: str, sample_pages: int = 5, min_chars: int = 100) -> bool:
    """Check if a PDF has embedded text (not scanned/image-only).

    Args:
        pdf_path: Path to PDF file.
        sample_pages: Number of pages to sample.
        min_chars: Minimum average chars/page to be considered text-based.

    Returns:
        True if the PDF has sufficient embedded text.
    """
    doc = fitz.open(pdf_path)
    try:
        pages_to_check = min(sample_pages, len(doc))
        if pages_to_check == 0:
            return False
        total_chars = sum(len(doc[i].get_text("text")) for i in range(pages_to_check))
        avg_chars = total_chars / pages_to_check
        return avg_chars > min_chars
    finally:
        doc.close()


def extract_deposition(pdf_path: str, output_dir: str) -> dict:
    """Extract deposition with exact line-by-line mapping using PyMuPDF.

    For each page:
    - Groups spans by y-coordinate into visual lines
    - Separates margin line numbers (x0 < 100, digits 1-25) from content
    - Detects page markers, headers, footers
    - Produces .md and _citations.json with exact line mappings

    Args:
        pdf_path: Path to the deposition PDF.
        output_dir: Directory for output files.

    Returns:
        Dict with keys: md_path, citations_path, page_count, line_count
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = _normalize_stem(pdf_path.stem)
    doc = fitz.open(str(pdf_path))

    all_pages = []
    citations = {}
    md_lines = []

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_data = _process_page(page, page_idx)
            all_pages.append(page_data)

            transcript_page = page_data["transcript_page"]

            # Use PDF page as fallback if no transcript page markers detected
            effective_transcript_page = transcript_page if transcript_page is not None else (page_idx + 1)

            # Always insert PAGE marker using effective page number
            md_lines.append(f"\n[PAGE:{effective_transcript_page}]")

            for line_info in page_data["lines"]:
                line_num = line_info["line_num"]
                text = line_info["text"]

                if not text.strip():
                    continue

                # Build citation key
                if effective_transcript_page is not None and line_num is not None:
                    key = f"line_P{effective_transcript_page}_L{line_num}"
                    citations[key] = CitationData(
                        page=page_idx + 1,
                        line_start=line_num,
                        line_end=line_num,
                        type="transcript_line",
                    ).to_dict()
                    citations[key]["transcript_page"] = effective_transcript_page

                # Format markdown with line numbers for depositions
                if line_num is not None:
                    # Format: "1  Q  Question text" or "2  A  Answer text"
                    md_lines.append(f"{line_num:2d}  {text}")
                else:
                    md_lines.append(text)

    finally:
        doc.close()

    # Write markdown
    md_path = output_dir / f"{stem}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Write citations
    citations_path = output_dir / f"{stem}_citations.json"
    with open(citations_path, "w") as f:
        json.dump(citations, f, indent=2)

    total_lines = sum(
        1 for p in all_pages for li in p["lines"] if li["text"].strip()
    )
    logger.info(
        "PyMuPDF extraction: %d pages, %d content lines, %d citations",
        len(all_pages),
        total_lines,
        len(citations),
    )

    return {
        "md_path": str(md_path),
        "citations_path": str(citations_path),
        "page_count": len(all_pages),
        "line_count": total_lines,
        "citation_count": len(citations),
    }


def _normalize_stem(name: str) -> str:
    """Normalize a filename stem to lowercase with underscores."""
    result = name.lower().replace(" ", "_").replace("-", "_")
    result = re.sub(r"_+", "_", result)
    return result.strip("_")


def _process_page(page: fitz.Page, page_idx: int) -> dict:
    """Process a single PDF page into structured line data.

    Returns:
        Dict with keys: pdf_page, transcript_page, lines
        Each line: {line_num, text, is_qa, y}
    """
    blocks = page.get_text("dict")["blocks"]

    # Collect all spans with their coordinates
    all_spans = []
    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                if not text.strip():
                    continue
                bbox = span["bbox"]  # (x0, y0, x1, y1)
                all_spans.append({
                    "text": text,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "y_mid": (bbox[1] + bbox[3]) / 2,
                })

    if not all_spans:
        return {"pdf_page": page_idx + 1, "transcript_page": None, "lines": []}

    # Group spans by y-coordinate
    visual_lines = _group_by_y(all_spans)

    # Parse each visual line
    transcript_page = None
    content_lines = []

    for y_key, spans in sorted(visual_lines.items()):
        y_mid = spans[0]["y_mid"]

        # Skip footer area
        if y_mid > FOOTER_Y_THRESHOLD:
            continue

        # Skip CONFIDENTIAL header
        if y_mid < HEADER_Y_THRESHOLD:
            full_text = " ".join(s["text"] for s in spans)
            if "CONFIDENTIAL" in full_text.upper():
                continue

        # Detect "Page N" marker
        page_marker = _detect_page_marker(spans)
        if page_marker is not None:
            transcript_page = page_marker
            continue

        # Separate margin line numbers from content
        margin_spans = []
        content_spans = []
        for s in sorted(spans, key=lambda s: s["x0"]):
            if s["x0"] < MARGIN_X_THRESHOLD and re.match(r"^\s*\d{1,2}\s*$", s["text"]):
                num = int(s["text"].strip())
                if 1 <= num <= 25:
                    margin_spans.append(num)
                    continue
            content_spans.append(s)

        if not content_spans:
            continue

        # Build content text
        content_text = " ".join(s["text"] for s in content_spans)
        # Clean up excessive internal whitespace while preserving Q/A indent
        content_text = re.sub(r"  +", "  ", content_text)

        line_num = margin_spans[0] if margin_spans else None

        # Detect Q/A markers
        is_qa = bool(re.match(r"\s*[QA]\s{2,}", content_text))

        content_lines.append({
            "line_num": line_num,
            "text": content_text,
            "is_qa": is_qa,
            "y": y_mid,
        })

    return {
        "pdf_page": page_idx + 1,
        "transcript_page": transcript_page,
        "lines": content_lines,
    }


def _group_by_y(spans: list) -> Dict[float, list]:
    """Group spans into visual lines by y-coordinate proximity.

    Spans within Y_GROUP_TOLERANCE of each other are considered the same line.
    Returns dict keyed by representative y value.
    """
    if not spans:
        return {}

    sorted_spans = sorted(spans, key=lambda s: s["y_mid"])
    groups: Dict[float, list] = {}
    current_y = sorted_spans[0]["y_mid"]
    current_group = [sorted_spans[0]]

    for span in sorted_spans[1:]:
        if abs(span["y_mid"] - current_y) <= Y_GROUP_TOLERANCE:
            current_group.append(span)
        else:
            groups[current_y] = current_group
            current_y = span["y_mid"]
            current_group = [span]

    if current_group:
        groups[current_y] = current_group

    return groups


def _detect_page_marker(spans: list) -> Optional[int]:
    """Detect 'Page N' marker from a visual line's spans.

    Page markers appear at x0 > 400 and y < 95 (near top-right).
    Also handles alternative format: "Conducted on ... N"
    """
    full_text = " ".join(s["text"] for s in spans).strip()

    # Standard format: "Page N"
    match = re.match(r"^Page\s+(\d+)$", full_text)
    if match:
        # Matched standard format
        pass
    else:
        # Alternative format: "Conducted on May 19, 2025 10" - extract final number after year
        match = re.search(r"Conducted on.*,\s*\d{4}\s+(\d{1,3})\s*$", full_text)
    if not match:
        return None

    # Check position requirements based on format type
    is_standard_format = re.match(r"^Page\s+\d+$", full_text) is not None
    is_conducted_format = "Conducted on" in full_text

    if is_standard_format:
        # Standard "Page N" - require specific position
        x0 = min(s["x0"] for s in spans)
        y_mid = spans[0]["y_mid"]
        if x0 > PAGE_MARKER_X_THRESHOLD and y_mid < PAGE_MARKER_Y_THRESHOLD:
            return int(match.group(1))
    elif is_conducted_format:
        # "Conducted on..." format - text pattern is specific enough, don't require position
        # This catches Planet Depos and similar court reporter formats
        return int(match.group(1))

    return None
