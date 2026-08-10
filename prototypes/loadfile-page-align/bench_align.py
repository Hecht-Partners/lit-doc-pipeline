#!/usr/bin/env python3
"""Ground-truth benchmark: take docs whose vendor text has form feeds matching
the page count, STRIP the form feeds, realign from per-page Tesseract OCR, and
score inferred pages against the vendor's true pages (token Jaccard)."""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from page_align_proto import parse_dat, parse_opt, resolve, LOADFILES, PROD_ROOT
from page_align_v2 import align_doc, ocr_page

TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
OUT = Path(__file__).parent / "bench_report.json"


def jaccard(a: str, b: str) -> float:
    sa, sb = set(TOKEN_RE.findall(a.lower())), set(TOKEN_RE.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def bench_doc(rec, tifs, prod_dir) -> dict | None:
    tp = resolve(rec["textfile"], prod_dir)
    vendor = tp.read_text(encoding="utf-8-sig", errors="replace")
    true_pages = vendor.split("\f")
    if len(true_pages) != rec["pages"]:
        return None
    stripped = vendor.replace("\f", "\n")
    tif_paths = [resolve(t, prod_dir) for t in tifs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        page_ocr = list(ex.map(ocr_page, tif_paths))
    boundaries, stats = align_doc(stripped, page_ocr)
    pieces, prev = [], 0
    for b in boundaries:
        pieces.append(stripped[prev:b]); prev = b
    pieces.append(stripped[prev:])
    scores = [round(jaccard(p, t), 3) for p, t in zip(pieces, true_pages)]
    return {
        "bates": rec["beg"], "pages": rec["pages"], "type": rec["rectype"],
        "filename": rec["filename"][:50], "scores": scores,
        "mean": round(sum(scores) / len(scores), 3),
        "pct_ge_090": round(sum(s >= 0.90 for s in scores) / len(scores), 3),
        "pct_ge_070": round(sum(s >= 0.70 for s in scores) / len(scores), 3),
    }


def main() -> int:
    budget_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    recs = parse_dat(LOADFILES / "REPROD001.dat")
    opt = parse_opt(LOADFILES / "REPROD001.opt")
    prod_dir = PROD_ROOT / "REPROD001"

    # choose ff-exact docs: mix of small/medium/large, email vs edoc vs attach
    chosen, used = [], 0
    buckets = {(lo, hi): 0 for lo, hi in [(2, 4), (5, 15), (16, 60), (61, 400)]}
    for r in recs:
        if r["pages"] < 2 or used + r["pages"] > budget_pages:
            continue
        tp = resolve(r["textfile"], prod_dir)
        ff = tp.read_text(encoding="utf-8-sig", errors="replace").count("\f")
        if ff + 1 != r["pages"]:
            continue
        for (lo, hi) in buckets:
            if lo <= r["pages"] <= hi and buckets[(lo, hi)] < 8:
                buckets[(lo, hi)] += 1
                chosen.append(r); used += r["pages"]
                break
    print(f"benchmarking {len(chosen)} docs, {used} pages", flush=True)

    results = []
    for r in chosen:
        print(f"  {r['beg']} ({r['pages']}p) ...", flush=True)
        out = bench_doc(r, opt[r["beg"]], prod_dir)
        if out:
            results.append(out)

    OUT.write_text(json.dumps(results, indent=2))
    tot = sum(r["pages"] for r in results)
    w = lambda key: sum(r[key] * r["pages"] for r in results) / tot
    print(f"\nDOCS={len(results)} PAGES={tot}")
    print(f"weighted mean jaccard={w('mean'):.3f}  pages>=0.90={w('pct_ge_090'):.1%}  pages>=0.70={w('pct_ge_070'):.1%}")
    worst = sorted(results, key=lambda r: r["mean"])[:6]
    for r in worst:
        print(f"  worst: {r['bates']} {r['pages']}p mean={r['mean']} {r['type']} {r['filename']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
