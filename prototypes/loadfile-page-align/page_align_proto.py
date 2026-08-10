#!/usr/bin/env python3
"""Prototype: place page boundaries (form feeds) in vendor doc-level text by
aligning it against fast per-page Tesseract OCR of the production TIFs.

Output: <out>/<bates>.paged.txt (vendor text + \f at page boundaries) and
report.json with per-page alignment confidence. The paged text is exactly what
lit_pipeline/loadfile_ingest.py consumes to emit per-page Bates citations.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

PROD_ROOT = Path("/Users/maximprice/Desktop/relator - ilinks/ilinks prod docs")
LOADFILES = PROD_ROOT / "load files"
OUT = Path(__file__).parent / "out"

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def parse_dat(path: Path) -> list[dict]:
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    hdr = [f.strip("\xfe") for f in rows[0].split("\x14")]
    idx = {n: i for i, n in enumerate(hdr)}

    def col(f, name):
        i = idx.get(name)
        return f[i].strip("\xfe") if i is not None and i < len(f) else ""

    recs = []
    for r in rows[1:]:
        if not r.strip():
            continue
        f = r.split("\x14")
        recs.append({
            "beg": col(f, "BEGDOC"), "end": col(f, "ENDDOC"),
            "pages": int(col(f, "PGCOUNT") or 1),
            "textfile": col(f, "TEXTFILE"), "native": col(f, "NATIVEFILE"),
            "filename": col(f, "FILENAME"), "rectype": col(f, "RECORD TYPE"),
        })
    return recs


def parse_opt(path: Path) -> dict[str, list[str]]:
    """Return beg_bates -> ordered list of TIF relative paths per document."""
    docs: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        bates, _, tif, docbreak = parts[0], parts[1], parts[2], parts[3]
        if docbreak.strip().upper() == "Y":
            current = bates
            docs[current] = []
        if current is not None:
            docs[current].append(tif)
    return docs


def resolve(rel: str, prod_dir: Path) -> Path:
    """Resolve a Windows relative path against the production dir or its parent."""
    rel = rel.lstrip(".\\").replace("\\", "/")
    for base in (prod_dir.parent, prod_dir):
        p = base / rel
        if p.exists():
            return p
    return prod_dir.parent / rel  # best guess; caller checks .exists()


def ocr_page(tif: Path) -> str:
    try:
        r = subprocess.run(
            ["tesseract", str(tif), "stdout", "-l", "eng", "--psm", "3"],
            capture_output=True, text=True, timeout=120,
        )
        return r.stdout or ""
    except Exception:
        return ""


def tokens_with_pos(text: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in TOKEN_RE.finditer(text.lower())]


def align_doc(vendor_text: str, page_ocr: list[str]) -> tuple[list[int], list[dict]]:
    """Return (boundary char offsets for pages 2..N, per-page stats)."""
    vt = tokens_with_pos(vendor_text)
    vtoks = [t for t, _ in vt]
    n_pages = len(page_ocr)
    cursor = 0
    boundaries: list[int] = []
    stats: list[dict] = []

    for k, ocr in enumerate(page_ocr):
        ptoks = TOKEN_RE.findall(ocr.lower())
        last_page = k == n_pages - 1
        if not ptoks:
            # blank/image-only page: zero-length segment, keep cursor
            stats.append({"page": k + 1, "ocr_tokens": 0, "matched": 0,
                          "conf": None, "note": "blank-ocr"})
            if not last_page:
                boundaries.append(vt[cursor][1] if cursor < len(vt) else len(vendor_text))
            continue

        win_end = min(len(vtoks), cursor + max(int(len(ptoks) * 1.9), len(ptoks) + 60))
        window = vtoks[cursor:win_end]
        sm = SequenceMatcher(None, window, ptoks, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        matched = sum(b.size for b in blocks)
        conf = matched / len(ptoks)
        if blocks:
            end_in_window = max(b.a + b.size for b in blocks)
        else:
            end_in_window = min(len(ptoks), len(window))
        stats.append({"page": k + 1, "ocr_tokens": len(ptoks), "matched": matched,
                      "conf": round(conf, 3)})
        if last_page:
            break
        cursor = min(cursor + end_in_window, len(vt))
        boundaries.append(vt[cursor][1] if cursor < len(vt) else len(vendor_text))

    return boundaries, stats


def process_doc(rec: dict, tifs: list[str], prod_dir: Path, workers: int = 8) -> dict:
    text_path = resolve(rec["textfile"], prod_dir)
    if not text_path.exists():
        return {"bates": rec["beg"], "status": "no-text"}
    vendor_text = text_path.read_text(encoding="utf-8-sig", errors="replace")

    if rec["pages"] == 1 or len(tifs) <= 1:
        out_text = vendor_text
        stats = [{"page": 1, "conf": 1.0, "note": "single-page (trivial)"}]
        boundaries = []
    else:
        tif_paths = [resolve(t, prod_dir) for t in tifs]
        missing = [p for p in tif_paths if not p.exists()]
        if missing:
            return {"bates": rec["beg"], "status": f"missing-tifs:{len(missing)}"}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            page_ocr = list(ex.map(ocr_page, tif_paths))
        boundaries, stats = align_doc(vendor_text, page_ocr)
        pieces, prev = [], 0
        for b in boundaries:
            pieces.append(vendor_text[prev:b])
            prev = b
        pieces.append(vendor_text[prev:])
        out_text = "\f".join(pieces)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{rec['beg']}.paged.txt").write_text(out_text, encoding="utf-8")
    confs = [s["conf"] for s in stats if s.get("conf") is not None]
    return {
        "bates": rec["beg"], "status": "ok", "pages": rec["pages"],
        "filename": rec["filename"], "boundaries_placed": len(boundaries),
        "min_conf": min(confs) if confs else None,
        "mean_conf": round(sum(confs) / len(confs), 3) if confs else None,
        "low_conf_pages": [s["page"] for s in stats if s.get("conf") is not None and s["conf"] < 0.35],
        "page_stats": stats,
    }


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
        print(f"{r['bates']:>14}  {r.get('pages','?'):>5}p  status={r['status']}"
              f"  min={r.get('min_conf')}  mean={r.get('mean_conf')}"
              f"  low_conf_pages={r.get('low_conf_pages')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
