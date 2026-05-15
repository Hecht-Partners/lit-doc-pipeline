"""
Plain-text court reporter transcript extractor.

Handles .txt deposition and hearing transcripts with standard formatting:
- Right-aligned page numbers (e.g., "                              42")
- Left-margin line numbers 1-25 (e.g., " 1   Q.  Did you review the document?")
- Q./A. markers for examination lines

Produces the same output format as pymupdf_extractor.py:
- .md with [PAGE:N] markers and line-numbered content
- _citations.json with line_P{page}_L{line} keys

Downstream chunking, indexing, and enrichment work unchanged.
"""

import json
import logging
import re
from pathlib import Path

from lit_pipeline.citation_types import CitationData

logger = logging.getLogger(__name__)

# Patterns for transcript detection
_LINE_NUM_RE = re.compile(r"^\s?(\d{1,2})\s{2,}")
_PAGE_NUM_RE = re.compile(r"^\s{30,}(\d+)\s*$")

# Pattern for Q./A. markers — strip trailing dot for chunker compatibility
_QA_DOT_RE = re.compile(r"^(\s*[QA])\.\s+", re.MULTILINE)


def is_text_transcript(txt_path: str) -> bool:
    """Check if a .txt file is a court reporter transcript.

    Reads the first ~80 lines and looks for both left-margin line numbers
    and right-aligned page numbers — the two hallmarks of a standard
    court reporter transcript.

    Args:
        txt_path: Path to a .txt file.

    Returns:
        True if the file looks like a court reporter transcript.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            head = [f.readline() for _ in range(80)]
    except (OSError, IOError):
        return False

    has_line_nums = any(_LINE_NUM_RE.match(line) for line in head if line)
    has_page_nums = any(_PAGE_NUM_RE.match(line) for line in head if line)
    return has_line_nums and has_page_nums


def extract_text_deposition(txt_path: str, output_dir: str) -> dict:
    """Extract a plain-text deposition transcript into .md + _citations.json.

    Parsing strategy:
    1. Split the file into pages at right-aligned page numbers.
    2. Within each page, extract left-margin line numbers and content.
    3. Strip Q./A. dots so the chunker's regex (``^\\d+\\s+[QA]\\s``) matches.

    Args:
        txt_path: Path to the .txt transcript.
        output_dir: Directory for output files (.md and _citations.json).

    Returns:
        Dict with keys: md_path, citations_path, page_count, line_count,
        citation_count.
    """
    txt_path = Path(txt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = _normalize_stem(txt_path.stem)

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # ── Pass 1: identify page boundaries ─────────────────────────────
    pages: list[dict] = []  # [{page_num, start_idx, end_idx}]
    for idx, line in enumerate(raw_lines):
        m = _PAGE_NUM_RE.match(line)
        if m:
            page_num = int(m.group(1))
            pages.append({"page_num": page_num, "start_idx": idx})

    # Set end indices
    for i in range(len(pages) - 1):
        pages[i]["end_idx"] = pages[i + 1]["start_idx"]
    if pages:
        pages[-1]["end_idx"] = len(raw_lines)

    # ── Pass 2: extract content lines per page ───────────────────────
    md_lines: list[str] = []
    citations: dict = {}
    total_content_lines = 0

    for page in pages:
        page_num = page["page_num"]
        md_lines.append(f"\n[PAGE:{page_num}]")

        for idx in range(page["start_idx"] + 1, page["end_idx"]):
            line = raw_lines[idx]
            m = _LINE_NUM_RE.match(line)
            if not m:
                continue

            line_num = int(m.group(1))
            if not (1 <= line_num <= 25):
                continue

            content = line[m.end():]
            content = content.rstrip("\n\r")

            # Strip Q./A. dots: "Q.  Did you" -> "Q  Did you"
            content = _QA_DOT_RE.sub(r"\1  ", content)

            if not content.strip():
                continue

            total_content_lines += 1

            # Citation key matches PyMuPDF format
            key = f"line_P{page_num}_L{line_num}"
            citations[key] = CitationData(
                page=page_num,
                line_start=line_num,
                line_end=line_num,
                type="transcript_line",
            ).to_dict()
            citations[key]["transcript_page"] = page_num

            md_lines.append(f"{line_num:2d}  {content}")

    # ── Write outputs ────────────────────────────────────────────────
    md_path = output_dir / f"{stem}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    citations_path = output_dir / f"{stem}_citations.json"
    with open(citations_path, "w") as f:
        json.dump(citations, f, indent=2)

    logger.info(
        "TextExtractor: %d pages, %d content lines, %d citations",
        len(pages),
        total_content_lines,
        len(citations),
    )

    return {
        "md_path": str(md_path),
        "citations_path": str(citations_path),
        "page_count": len(pages),
        "line_count": total_content_lines,
        "citation_count": len(citations),
    }


def _normalize_stem(name: str) -> str:
    """Normalize a filename stem to lowercase with underscores."""
    result = name.lower().replace(" ", "_").replace("-", "_")
    result = re.sub(r"_+", "_", result)
    return result.strip("_")
