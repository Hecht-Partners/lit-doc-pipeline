#!/usr/bin/env python3
"""Objective check of paged output: for each page k, the tokens of text
segment k should match page k's OCR better than the neighboring pages' OCR.
Reports diagonal-win rate per document."""

import re
import sys
from pathlib import Path

from page_align_proto import parse_dat, parse_opt, PROD_ROOT, LOADFILES

OUT = Path(__file__).parent / "out_v2"
CACHE = Path(__file__).parent / "ocr_cache"
TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def toks(s: str) -> set[str]:
    return set(TOKEN_RE.findall(s.lower()))


def containment(seg: set[str], page: set[str]) -> float:
    if not seg or not page:
        return 0.0
    return len(seg & page) / min(len(seg), len(page))


def main() -> int:
    for prod in ("REPROD001", "PROD004"):
        recs = {r["beg"]: r for r in parse_dat(LOADFILES / f"{prod}.dat")}
        opt = parse_opt(LOADFILES / f"{prod}.opt")
        for bid in sys.argv[1:]:
            rec = recs.get(bid)
            if not rec or rec["pages"] < 2:
                continue
            paged = OUT / f"{bid}.paged.txt"
            if not paged.exists():
                continue
            segs = paged.read_text(encoding="utf-8").split("\f")
            tifs = opt[bid]
            ocr = [toks((CACHE / (Path(t).stem + ".txt")).read_text(errors="replace"))
                   if (CACHE / (Path(t).stem + ".txt")).exists() else set()
                   for t in [p.replace("\\", "/") for p in tifs]]
            n = len(segs)
            wins = ties = losses = empty = 0
            bad_pages = []
            for k in range(n):
                seg = toks(segs[k])
                if not seg or not ocr[k]:
                    empty += 1
                    continue
                own = containment(seg, ocr[k])
                rivals = [containment(seg, ocr[j])
                          for j in (k - 1, k + 1) if 0 <= j < n]
                if own >= max(rivals, default=0) and own > 0.3:
                    wins += 1
                elif own >= max(rivals, default=0):
                    ties += 1
                else:
                    losses += 1
                    bad_pages.append((k + 1, round(own, 2), round(max(rivals), 2)))
            print(f"{bid:>14} {n:>4}p  correct={wins}  weak={ties}  wrong={losses}"
                  f"  empty={empty}  wrong_pages={bad_pages[:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
