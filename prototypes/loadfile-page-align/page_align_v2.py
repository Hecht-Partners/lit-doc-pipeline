#!/usr/bin/env python3
"""Prototype v2: global anchor-based page alignment.

Instead of a greedy per-page cursor (v1, cascaded on scanned docs), collect
globally-unique token/bigram anchors linking vendor-text positions to pages,
enforce monotonicity via longest non-decreasing subsequence, and place page
boundaries between the last anchor of page k and the first anchor of page k+1
(midpoint, snapped to a line break). Pages without anchors are interpolated
and flagged.
"""

from __future__ import annotations

import bisect
import json
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from page_align_proto import parse_dat, parse_opt, resolve, PROD_ROOT, LOADFILES

OUT = Path(__file__).parent / "out_v2"
TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


OCR_CACHE = Path(__file__).parent / "ocr_cache"


def ocr_page(tif: Path) -> str:
    OCR_CACHE.mkdir(exist_ok=True)
    cache = OCR_CACHE / (tif.stem + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    try:
        r = subprocess.run(
            ["tesseract", str(tif), "stdout", "-l", "eng", "--psm", "3"],
            capture_output=True, text=True, timeout=120,
        )
        out = r.stdout or ""
    except Exception:
        out = ""
    cache.write_text(out, encoding="utf-8")
    return out


def lis_by_page(anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """anchors sorted by vendor token idx; keep longest non-decreasing page
    subsequence (patience sorting, O(n log n))."""
    tails: list[int] = []          # tails[i] = smallest page ending a subseq of len i+1
    tails_idx: list[int] = []
    prev = [-1] * len(anchors)
    for i, (_, page) in enumerate(anchors):
        j = bisect.bisect_right(tails, page)
        if j == len(tails):
            tails.append(page); tails_idx.append(i)
        else:
            tails[j] = page; tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    out = []
    i = tails_idx[-1] if tails_idx else -1
    while i >= 0:
        out.append(anchors[i]); i = prev[i]
    return out[::-1]


def align_doc(vendor_text: str, page_ocr: list[str]) -> tuple[list[int], dict]:
    vlow = vendor_text.lower()
    vt = [(m.group(0), m.start()) for m in TOKEN_RE.finditer(vlow)]
    vtoks = [t for t, _ in vt]
    n_pages = len(page_ocr)

    # Vendor indexes: unique single tokens and unique consecutive bigrams
    tok_pos = defaultdict(list)
    for i, t in enumerate(vtoks):
        tok_pos[t].append(i)
    bi_pos = defaultdict(list)
    for i in range(len(vtoks) - 1):
        bi_pos[(vtoks[i], vtoks[i + 1])].append(i)

    # Page OCR tokens; track which pages each token appears on (to require
    # page-uniqueness too)
    page_toks = [TOKEN_RE.findall(o.lower()) for o in page_ocr]
    tok_pages = defaultdict(set)
    for k, toks in enumerate(page_toks):
        for t in toks:
            tok_pages[t].add(k)

    anchors: list[tuple[int, int]] = []  # (vendor token idx, page)
    seen = set()
    for k, toks in enumerate(page_toks):
        for a, b in zip(toks, toks[1:]):
            hits = bi_pos.get((a, b))
            if hits and len(hits) == 1 and (hits[0], k) not in seen:
                anchors.append((hits[0], k)); seen.add((hits[0], k))
        for t in set(toks):
            if len(t) >= 5 and len(tok_pos.get(t, ())) == 1 and len(tok_pages[t]) == 1:
                key = (tok_pos[t][0], k)
                if key not in seen:
                    anchors.append(key); seen.add(key)

    anchors.sort()
    kept = lis_by_page(anchors) if anchors else []

    per_page_anchors = defaultdict(list)
    for idx, page in kept:
        per_page_anchors[page].append(idx)

    from difflib import SequenceMatcher

    def refine(lo: int, hi: int, next_page: int) -> tuple[int | None, float]:
        """Locate where page `next_page` begins inside vendor tokens [lo, hi)
        by matching the head of its OCR tokens. Returns (token idx, ratio)."""
        head = page_toks[next_page][:60]
        window = vtoks[lo:hi]
        if not head or not window:
            return None, 0.0
        sm = SequenceMatcher(None, window, head, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
        if not blocks:
            return None, 0.0
        matched = sum(b.size for b in blocks)
        # start of page = window position of first solid match, backed off by
        # how far into the OCR head that match starts
        first = blocks[0]
        return lo + max(first.a - first.b, 0), matched / len(head)

    # Boundary for page transition k -> k+1 (n_pages-1 boundaries)
    boundaries: list[int] = []
    gaps: list[dict] = []
    last_known = 0
    for k in range(n_pages - 1):
        left = max((max(per_page_anchors[p]) for p in range(k + 1) if per_page_anchors[p]), default=None)
        right = min((min(per_page_anchors[p]) for p in range(k + 1, n_pages) if per_page_anchors[p]), default=None)
        refined = False
        if left is not None and right is not None and right >= left:
            gap = right - left
            if gap > 40:  # anchors far apart: locate page start within the gap
                mid_tok, ratio = refine(left, right + 1, k + 1)
                if mid_tok is not None and ratio >= 0.3:
                    refined = True
                    gap = 0 if ratio >= 0.5 else gap
                else:
                    mid_tok = (left + right + 1) // 2
            else:
                mid_tok = (left + right + 1) // 2
        elif right is not None:
            mid_tok, gap = right, -1
        elif left is not None:
            mid_tok, gap = min(left + 1, len(vt)), -1
        else:
            mid_tok, gap = last_known, -1
        mid_tok = max(mid_tok, last_known)  # keep boundaries monotonic
        last_known = mid_tok
        char = vt[mid_tok][1] if mid_tok < len(vt) else len(vendor_text)
        nl = vendor_text.rfind("\n", 0, char)
        boundaries.append(nl + 1 if nl != -1 else char)
        gaps.append({"after_page": k + 1, "gap_tokens": gap, "refined": refined})

    # normalize boundary monotonicity at char level
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1])

    stats = {
        "anchors_total": len(anchors), "anchors_kept": len(kept),
        "pages_with_anchors": sum(1 for p in range(n_pages) if per_page_anchors[p]),
        "n_pages": n_pages,
        "anchorless_pages": [p + 1 for p in range(n_pages) if not per_page_anchors[p]],
        "wide_boundaries": [g for g in gaps if g["gap_tokens"] > 80 or g["gap_tokens"] < 0],
    }
    return boundaries, stats


def process_doc(rec, tifs, prod_dir, workers=8):
    text_path = resolve(rec["textfile"], prod_dir)
    if not text_path.exists():
        return {"bates": rec["beg"], "status": "no-text"}
    vendor_text = text_path.read_text(encoding="utf-8-sig", errors="replace")

    if rec["pages"] == 1 or len(tifs) <= 1:
        out_text, stats = vendor_text, {"n_pages": 1, "note": "single-page"}
    else:
        tif_paths = [resolve(t, prod_dir) for t in tifs]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            page_ocr = list(ex.map(ocr_page, tif_paths))
        boundaries, stats = align_doc(vendor_text, page_ocr)
        pieces, prev = [], 0
        for b in boundaries:
            pieces.append(vendor_text[prev:b]); prev = b
        pieces.append(vendor_text[prev:])
        out_text = "\f".join(pieces)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{rec['beg']}.paged.txt").write_text(out_text, encoding="utf-8")
    return {"bates": rec["beg"], "status": "ok", "pages": rec["pages"],
            "filename": rec["filename"], **stats}


def main() -> int:
    sample_ids = sys.argv[1:]
    results = []
    for prod, dirname in [("REPROD001", "REPROD001"), ("PROD004", "PROD004")]:
        recs = {r["beg"]: r for r in parse_dat(LOADFILES / f"{prod}.dat")}
        opt = parse_opt(LOADFILES / f"{prod}.opt")
        prod_dir = PROD_ROOT / dirname
        for bid in sample_ids:
            if bid in recs:
                print(f"processing {bid} ({recs[bid]['pages']}p) ...", flush=True)
                results.append(process_doc(recs[bid], opt.get(bid, []), prod_dir))
    (OUT / "report.json").write_text(json.dumps(results, indent=2))
    for r in results:
        if r.get("n_pages", 1) > 1:
            print(f"{r['bates']:>14} {r['pages']:>5}p anchors={r['anchors_kept']}/{r['anchors_total']}"
                  f" pages_anchored={r['pages_with_anchors']}/{r['n_pages']}"
                  f" anchorless={r['anchorless_pages'][:8]}{'...' if len(r['anchorless_pages'])>8 else ''}"
                  f" wide={len(r['wide_boundaries'])}")
        else:
            print(f"{r['bates']:>14} {r.get('pages','?'):>5}p single-page (exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
