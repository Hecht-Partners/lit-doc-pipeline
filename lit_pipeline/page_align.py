"""
Page-boundary recovery for load-file productions whose vendor text lacks
form-feed page breaks.

Vendor TEXT exports are document-level; when they carry form feeds that match
the production's per-document page count (PGCOUNT / Bates-range width), those
are authoritative and used as-is. When they don't, this module recovers page
boundaries by aligning the vendor text against fast per-page Tesseract OCR of
the production TIFs:

1. OCR each page TIF (fingerprint only — the vendor text remains the citable
   text, so OCR noise never enters the corpus).
2. Collect anchors: tokens and consecutive token bigrams that occur exactly
   once in the vendor text and on exactly one page's OCR.
3. Enforce monotonicity with a longest non-decreasing subsequence over page
   order (a garbled page cannot cascade into its neighbors).
4. Place each boundary between the last anchor of page k and the first anchor
   of page k+1; when that gap is wide, locally match the head of page k+1's
   OCR inside the anchor-bounded window to tighten it.

Benchmarked against vendor ground truth (documents whose text DID carry form
feeds, stripped and realigned from OCR): weighted mean page Jaccard 0.982,
95.9% of pages >= 0.90 (700 scanned tax-document pages, 2026-08-10).

Citation-safety contract: callers receive per-document routing and confidence
info. Documents routed 'aligned' with poor confidence, and documents whose
form-feed count contradicts the produced page count (text extracted from a
native with different pagination, e.g. 4-up W2 sheets), are flagged so
downstream cites fall back to the document's Bates range instead of a guessed
page.
"""

from __future__ import annotations

import bisect
import logging
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Anchor gaps wider than this many vendor tokens trigger local refinement;
# refined boundaries with a match ratio below _REFINE_MIN_RATIO keep the
# midpoint guess and count as wide.
_WIDE_GAP_TOKENS = 40
_REFINE_MIN_RATIO = 0.3
_REFINE_GOOD_RATIO = 0.5
_OCR_HEAD_TOKENS = 60

# A doc is flagged when more than this fraction of its boundaries stay wide
# or of its pages have no anchors at all.
_FLAG_FRACTION = 0.10


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page(tif_path: Path, cache_dir: Optional[Path] = None,
             timeout: int = 120) -> str:
    """Fast single-page OCR for alignment fingerprints (cached by TIF stem)."""
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (tif_path.stem + ".txt")
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8", errors="replace")
    try:
        result = subprocess.run(
            ["tesseract", str(tif_path), "stdout", "-l", "eng", "--psm", "3"],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout or ""
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Tesseract failed on %s: %s", tif_path.name, e)
        out = ""
    if cache_file is not None:
        cache_file.write_text(out, encoding="utf-8")
    return out


@dataclass
class AlignmentStats:
    n_pages: int
    anchors_total: int = 0
    anchors_kept: int = 0
    pages_with_anchors: int = 0
    anchorless_pages: list[int] = field(default_factory=list)
    wide_boundaries: int = 0

    @property
    def flagged(self) -> bool:
        if self.n_pages <= 1:
            return False
        n_bounds = self.n_pages - 1
        return (self.wide_boundaries > _FLAG_FRACTION * n_bounds
                or len(self.anchorless_pages) > _FLAG_FRACTION * self.n_pages)


def _lis_by_page(anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest non-decreasing subsequence of pages over anchors sorted by
    vendor position (patience sorting, O(n log n))."""
    tails: list[int] = []
    tails_idx: list[int] = []
    prev = [-1] * len(anchors)
    for i, (_, page) in enumerate(anchors):
        j = bisect.bisect_right(tails, page)
        if j == len(tails):
            tails.append(page)
            tails_idx.append(i)
        else:
            tails[j] = page
            tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    out: list[tuple[int, int]] = []
    i = tails_idx[-1] if tails_idx else -1
    while i >= 0:
        out.append(anchors[i])
        i = prev[i]
    return out[::-1]


def align_pages(vendor_text: str, page_ocr: list[str]) -> tuple[list[int], AlignmentStats]:
    """Return (char offsets where pages 2..N begin, stats).

    Pure function: page_ocr is one OCR string per page, in page order.
    """
    n_pages = len(page_ocr)
    stats = AlignmentStats(n_pages=n_pages)
    if n_pages <= 1:
        return [], stats

    vlow = vendor_text.lower()
    vt = [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(vlow)]
    vtoks = [t for t, _ in vt]

    tok_pos: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(vtoks):
        tok_pos[t].append(i)
    bi_pos: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(vtoks) - 1):
        bi_pos[(vtoks[i], vtoks[i + 1])].append(i)

    page_toks = [_TOKEN_RE.findall(o.lower()) for o in page_ocr]
    tok_pages: dict[str, set[int]] = defaultdict(set)
    for k, toks in enumerate(page_toks):
        for t in toks:
            tok_pages[t].add(k)

    anchors: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for k, toks in enumerate(page_toks):
        for a, b in zip(toks, toks[1:]):
            hits = bi_pos.get((a, b))
            if hits and len(hits) == 1 and (hits[0], k) not in seen:
                anchors.append((hits[0], k))
                seen.add((hits[0], k))
        for t in set(toks):
            if len(t) >= 5 and len(tok_pos.get(t, ())) == 1 and len(tok_pages[t]) == 1:
                key = (tok_pos[t][0], k)
                if key not in seen:
                    anchors.append(key)
                    seen.add(key)

    anchors.sort()
    kept = _lis_by_page(anchors) if anchors else []

    per_page: dict[int, list[int]] = defaultdict(list)
    for idx, page in kept:
        per_page[page].append(idx)

    stats.anchors_total = len(anchors)
    stats.anchors_kept = len(kept)
    stats.pages_with_anchors = sum(1 for p in range(n_pages) if per_page[p])
    stats.anchorless_pages = [p + 1 for p in range(n_pages) if not per_page[p]]

    def refine(lo: int, hi: int, next_page: int) -> tuple[Optional[int], float]:
        head = page_toks[next_page][:_OCR_HEAD_TOKENS]
        window = vtoks[lo:hi]
        if not head or not window:
            return None, 0.0
        sm = SequenceMatcher(None, window, head, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
        if not blocks:
            return None, 0.0
        matched = sum(b.size for b in blocks)
        first = blocks[0]
        return lo + max(first.a - first.b, 0), matched / len(head)

    boundaries: list[int] = []
    last_known = 0
    for k in range(n_pages - 1):
        left = max((max(per_page[p]) for p in range(k + 1) if per_page[p]),
                   default=None)
        right = min((min(per_page[p]) for p in range(k + 1, n_pages) if per_page[p]),
                    default=None)
        wide = False
        if left is not None and right is not None and right >= left:
            gap = right - left
            if gap > _WIDE_GAP_TOKENS:
                mid_tok, ratio = refine(left, right + 1, k + 1)
                if mid_tok is None or ratio < _REFINE_MIN_RATIO:
                    mid_tok = (left + right + 1) // 2
                    wide = True
                elif ratio < _REFINE_GOOD_RATIO:
                    wide = True
            else:
                mid_tok = (left + right + 1) // 2
        elif right is not None:
            mid_tok, wide = right, True
        elif left is not None:
            mid_tok, wide = min(left + 1, len(vt)), True
        else:
            mid_tok, wide = last_known, True
        mid_tok = max(mid_tok, last_known)
        last_known = mid_tok
        if wide:
            stats.wide_boundaries += 1
        char = vt[mid_tok][1] if mid_tok < len(vt) else len(vendor_text)
        nl = vendor_text.rfind("\n", 0, char)
        boundaries.append(nl + 1 if nl != -1 else char)

    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1])
    return boundaries, stats


@dataclass
class PaginationResult:
    text: str                 # vendor text with \f at page boundaries
    route: str                # single | vendor-ff | aligned | unpaged-mismatch | unpaged-no-ocr
    flagged: bool = False
    reason: str = ""
    stats: Optional[AlignmentStats] = None


def paginate_vendor_text(
    vendor_text: str,
    n_pages: int,
    tif_paths: Optional[list[Path]] = None,
    ocr_cache: Optional[Path] = None,
    ocr_workers: int = 8,
    ocr_fn: Optional[Callable[[Path], str]] = None,
) -> PaginationResult:
    """Segment document-level vendor text into n_pages using the best
    available evidence. Never fabricates page precision: mismatched or
    unalignable documents come back unpaged and flagged so cites fall back
    to the document's Bates range."""
    if n_pages <= 1:
        return PaginationResult(text=vendor_text.replace("\f", "\n"), route="single")

    segments = vendor_text.split("\f")
    # trailing empty segments (e.g. a terminal form feed) don't count as pages
    while len(segments) > n_pages and not segments[-1].strip():
        segments.pop()

    if len(segments) == n_pages:
        return PaginationResult(text="\f".join(segments), route="vendor-ff")

    if len(segments) > 1:
        return PaginationResult(
            text=vendor_text.replace("\f", "\n"), route="unpaged-mismatch",
            flagged=True,
            reason=(f"vendor text has {len(segments)} form-feed pages but the "
                    f"produced document has {n_pages} pages (text likely "
                    f"extracted from a native with different pagination)"),
        )

    if not tif_paths or len(tif_paths) != n_pages:
        return PaginationResult(
            text=vendor_text, route="unpaged-no-ocr", flagged=True,
            reason="no per-page images available for alignment",
        )
    if ocr_fn is None:
        if not tesseract_available():
            return PaginationResult(
                text=vendor_text, route="unpaged-no-ocr", flagged=True,
                reason="tesseract not installed; cannot align pages",
            )
        ocr_fn = lambda p: ocr_page(p, cache_dir=ocr_cache)  # noqa: E731

    with ThreadPoolExecutor(max_workers=ocr_workers) as ex:
        page_ocr = list(ex.map(ocr_fn, tif_paths))
    boundaries, stats = align_pages(vendor_text, page_ocr)

    pieces, prev = [], 0
    for b in boundaries:
        pieces.append(vendor_text[prev:b])
        prev = b
    pieces.append(vendor_text[prev:])
    return PaginationResult(
        text="\f".join(pieces), route="aligned", flagged=stats.flagged,
        reason=(f"alignment confidence low: {stats.wide_boundaries} wide "
                f"boundaries, {len(stats.anchorless_pages)} anchorless pages"
                if stats.flagged else ""),
        stats=stats,
    )


def parse_opt(opt_path: Path) -> dict[str, list[str]]:
    """Parse an Opticon .opt image cross-reference.

    Returns beg_bates -> ordered list of image relative paths (one per page).
    Format: BATES,VOLUME,PATH,DOCBREAK(Y|blank),FOLDERBREAK,BOXBREAK,PAGECOUNT
    """
    docs: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in opt_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        bates, _vol, image_path, doc_break = parts[0], parts[1], parts[2], parts[3]
        if doc_break.strip().upper() == "Y":
            current = bates.strip()
            docs[current] = []
        if current is not None:
            docs[current].append(image_path.strip())
    return docs
