"""
Load-file ingest adapter for Concordance-style productions.

Vendor productions (Relativity / Concordance) ship with extracted text already
in TEXT/. Routing those through Docling/OCR is wasteful and lower-fidelity than
the vendor text. This module writes converted/{bates}.md + _citations.json
directly into the corpus (same shape post_processor would produce), then runs
the chunker per document. The pipeline's `process` step is bypassed entirely;
the user runs `lit_pipeline.py index` after to rebuild BM25 + vector indexes
incrementally.

Routing per .dat record:
  - TEXT route: vendor TEXTPATH points to a non-empty text file → wrap as .md
    with [TEXT:N] paragraph markers, write companion _citations.json. Form
    feeds in the vendor text delimit pages: each paragraph is attributed its
    true page number, and when the PRODBEG–PRODEND range width matches the
    page count each page gets its own Bates stamp (image productions stamp
    one Bates per page).
  - Native route: TEXT is empty/missing but NATIVEFILE points to a
    format_handlers-supported native (xlsx/pptx/eml/msg) → run format_handlers
    to get .md, then post-process to insert markers + write _citations.json.
  - Skipped: neither usable → logged to flagged_no_content.json.

Concordance load-file format expected:
  - UTF-8 with optional BOM
  - Field delimiter: 0x14 (DC4)
  - Text qualifier: 0xfe (þ) — stripped from each field
  - Row delimiter: CRLF
  - Header row names: PRODBEG, PRODEND, PRODBEGATT, PRODENDATT, CUSTODIAN,
    NATIVEFILE, FILEDESC, FOLDER, FILENAME, DOCEXT, PAGES, AUTHOR, DATECREATED,
    DATELASTMOD, SUBJECT, FROM, TO, CC, BCC, DATESENT, DATERCVD,
    CONFIDENTIALITY, TEXTPATH, PRODVOL (vendor-extra fields ignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum text size (after stripping BOM/whitespace) to treat vendor text as
# usable. Empty/near-empty .txt files happen when the vendor produced an image
# without OCR. Set low enough to catch tabular XLSX exports (40-50 bytes of
# useful data); SharePoint wiki stubs below this threshold are usually empty
# placeholders.
MIN_TEXT_BYTES = 15

# Native extensions format_handlers can extract. Anything else (video, audio,
# OneNote TOCs, Visio diagrams, raster images) is flagged and skipped.
PARSEABLE_NATIVE_EXTS = {
    "docx", "doc", "xlsx", "xls", "pptx", "ppt",
    "msg", "eml", "txt", "md", "rtf", "html", "htm",
    "csv", "tsv", "json", "xml",
}


@dataclass
class DocRecord:
    """One row from the .dat load file."""
    beg_bates: str
    end_bates: str
    beg_attach: str
    end_attach: str
    custodian: str
    native_path: str
    file_desc: str
    folder: str
    filename: str
    doc_ext: str
    pages: str
    author: str
    date_created: str
    date_last_mod: str
    subject: str
    from_field: str
    to_field: str
    cc: str
    bcc: str
    date_sent: str
    date_rcvd: str
    confidentiality: str
    text_path: str
    prod_vol: str


# Vendors name the same columns differently. Canonical name -> accepted
# aliases, matched case-insensitively (first alias present in the header
# wins). Extend here when a new vendor vocabulary shows up.
HEADER_ALIASES: dict[str, list[str]] = {
    "PRODBEG": ["PRODBEG", "BEGDOC", "BEGBATES", "BATES BEG"],
    "PRODEND": ["PRODEND", "ENDDOC", "ENDBATES", "BATES END"],
    "PRODBEGATT": ["PRODBEGATT", "BEGATTACH", "BEGATT"],
    "PRODENDATT": ["PRODENDATT", "ENDATTACH", "ENDATT"],
    "CUSTODIAN": ["CUSTODIAN"],
    "NATIVEFILE": ["NATIVEFILE", "NATIVELINK", "NATIVEPATH"],
    "FILEDESC": ["FILEDESC", "RECORD TYPE", "RECORDTYPE"],
    "FOLDER": ["FOLDER", "FILE_PATH", "FILEPATH"],
    "FILENAME": ["FILENAME"],
    "DOCEXT": ["DOCEXT", "FILE EXTENSION", "FILEEXT"],
    "PAGES": ["PAGES", "PGCOUNT", "PAGECOUNT", "PAGE COUNT"],
    "AUTHOR": ["AUTHOR"],
    "DATECREATED": ["DATECREATED", "DATETIMECREATED"],
    "DATELASTMOD": ["DATELASTMOD", "DATE TIME MOD"],
    "SUBJECT": ["SUBJECT"],
    "FROM": ["FROM"],
    "TO": ["TO"],
    "CC": ["CC"],
    "BCC": ["BCC"],
    "DATESENT": ["DATESENT", "DATETIMESENT"],
    "DATERCVD": ["DATERCVD", "DATETIMERCVD"],
    "CONFIDENTIALITY": ["CONFIDENTIALITY"],
    "TEXTPATH": ["TEXTPATH", "TEXTFILE", "TEXTLINK", "OCRPATH", "FULLTEXT"],
    "PRODVOL": ["PRODVOL", "VOLUME", "SOURCE ID (BOX #)"],
}


def parse_dat(dat_path: Path) -> list[DocRecord]:
    """Parse a Concordance .dat into a list of DocRecord."""
    raw = dat_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8")
    rows = text.splitlines()
    if not rows:
        return []

    header = [f.strip("﻿").strip("þ").strip().upper() for f in rows[0].split("\x14")]
    raw_idx = {name: i for i, name in enumerate(header)}
    name_to_idx: dict[str, int] = {}
    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if alias in raw_idx:
                name_to_idx[canonical] = raw_idx[alias]
                break

    def col(fields: list[str], name: str) -> str:
        i = name_to_idx.get(name)
        if i is None or i >= len(fields):
            return ""
        return fields[i].strip("þ")

    records: list[DocRecord] = []
    for row in rows[1:]:
        if not row.strip():
            continue
        fields = row.split("\x14")
        records.append(
            DocRecord(
                beg_bates=col(fields, "PRODBEG"),
                end_bates=col(fields, "PRODEND"),
                beg_attach=col(fields, "PRODBEGATT"),
                end_attach=col(fields, "PRODENDATT"),
                custodian=col(fields, "CUSTODIAN"),
                native_path=col(fields, "NATIVEFILE"),
                file_desc=col(fields, "FILEDESC"),
                folder=col(fields, "FOLDER"),
                filename=col(fields, "FILENAME"),
                doc_ext=col(fields, "DOCEXT"),
                pages=col(fields, "PAGES"),
                author=col(fields, "AUTHOR"),
                date_created=col(fields, "DATECREATED"),
                date_last_mod=col(fields, "DATELASTMOD"),
                subject=col(fields, "SUBJECT"),
                from_field=col(fields, "FROM"),
                to_field=col(fields, "TO"),
                cc=col(fields, "CC"),
                bcc=col(fields, "BCC"),
                date_sent=col(fields, "DATESENT"),
                date_rcvd=col(fields, "DATERCVD"),
                confidentiality=col(fields, "CONFIDENTIALITY"),
                text_path=col(fields, "TEXTPATH"),
                prod_vol=col(fields, "PRODVOL"),
            )
        )
    return records


def _to_posix(rel: str) -> str:
    """Convert backslash-separated Windows relative path to POSIX."""
    rel = rel.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    return rel


def _resolution_bases(production_root: Path) -> list[Path]:
    """Candidate bases for load-file relative paths. Vendors write paths
    relative to the delivery root (`PROD004\\TEXT\\...`), which may sit at,
    above, or one level below the production dir (e.g. the doubled
    REPROD001/REPROD001/ layout)."""
    return [
        production_root,
        production_root.parent,
        production_root / production_root.name,
    ]


def _resolve_rel(rel: str, production_root: Path) -> Optional[Path]:
    """Resolve a load-file relative path against the known bases."""
    posix = _to_posix(rel)
    for base in _resolution_bases(production_root):
        p = base / posix
        if p.exists():
            return p
    return None


def _read_text_bytes(path: Path) -> bytes:
    """Read file bytes with BOM stripped, empty on error."""
    try:
        data = path.read_bytes()
    except OSError:
        return b""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data


def _is_usable_text(data: bytes) -> bool:
    """Return True if the text has enough non-whitespace content to chunk."""
    return len(data.translate(None, b" \t\r\n")) >= MIN_TEXT_BYTES


def _build_header(rec: DocRecord) -> list[str]:
    """Build a short human-readable header block for the .md."""
    lines = [
        f"# {rec.beg_bates}",
        "",
        f"**Bates range:** {rec.beg_bates} – {rec.end_bates}",
    ]
    if rec.filename:
        lines.append(f"**Original filename:** {rec.filename}")
    if rec.custodian:
        lines.append(f"**Custodian:** {rec.custodian}")
    date_line = rec.date_sent or rec.date_created or rec.date_last_mod
    if date_line:
        lines.append(f"**Date:** {date_line}")
    if rec.from_field:
        lines.append(f"**From:** {rec.from_field}")
    if rec.to_field:
        lines.append(f"**To:** {rec.to_field}")
    if rec.cc:
        lines.append(f"**Cc:** {rec.cc}")
    if rec.subject:
        lines.append(f"**Subject:** {rec.subject}")
    if rec.confidentiality:
        lines.append(f"**Confidentiality:** {rec.confidentiality}")
    return lines


def _split_paragraphs(text: str) -> list[str]:
    """Split body text into paragraphs on blank lines, preserving non-empty blocks."""
    # Collapse runs of blank lines into a single delimiter, then split.
    normalized = re.sub(r"(\r?\n){2,}", "\n\n", text).strip()
    if not normalized:
        return []
    return [p.strip() for p in normalized.split("\n\n") if p.strip()]


_BATES_NUM_RE = re.compile(r"^(.*?)(\d+)$")


def _parse_bates(bates: str) -> Optional[tuple[str, int, int]]:
    """Split a Bates stamp into (prefix, number, digit-width), or None."""
    m = _BATES_NUM_RE.match(bates.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), len(m.group(2))


def _split_vendor_pages(text: str) -> list[str]:
    """Split vendor text on form feeds — the page delimiter in Concordance
    TEXT exports. Returns one segment per page (segments may be empty for
    image-only pages; callers must preserve positions for page numbering)."""
    return text.split("\f") if "\f" in text else [text]


def _page_bates_list(rec: DocRecord, n_pages: int) -> list[str]:
    """Derive one Bates stamp per page from the record's PRODBEG–PRODEND range.

    Image productions stamp one Bates number per page, so when the range width
    equals the form-feed page count we can attribute each page its own stamp.
    Falls back to PRODBEG for every page when the range doesn't line up.
    """
    beg = _parse_bates(rec.beg_bates)
    end = _parse_bates(rec.end_bates or rec.beg_bates)
    if beg and end and beg[0] == end[0]:
        width = end[1] - beg[1] + 1
        if width == n_pages:
            prefix, num, pad = beg
            return [f"{prefix}{num + i:0{pad}d}" for i in range(n_pages)]
        if n_pages > 1:
            logger.warning(
                "%s: %d form-feed pages but Bates range spans %d — "
                "using PRODBEG for all pages",
                rec.beg_bates, n_pages, width,
            )
    return [rec.beg_bates] * n_pages


def _write_md_and_citations(
    rec: DocRecord,
    body_text: str,
    converted_dir: Path,
) -> tuple[Path, Path]:
    """Write converted/{bates}.md and converted/{bates}_citations.json.

    Inserts [TEXT:N] markers before each paragraph so the chunker can map them
    back to the bates anchor via #/texts/N keys in citations. Vendor text is
    split on form feeds into true pages; each paragraph's citation entry
    carries its real page number and (when the Bates range width matches the
    page count) that page's own Bates stamp, so the chunker's page-precision
    layer ([PAGE:N | BATES] markers, page_spans, exact citation_string pages)
    produces pincite-accurate chunks.
    """
    bates_lower = rec.beg_bates.lower()
    md_path = converted_dir / f"{bates_lower}.md"
    cite_path = converted_dir / f"{bates_lower}_citations.json"

    header_lines = _build_header(rec)
    page_segments = _split_vendor_pages(body_text)
    page_bates = _page_bates_list(rec, len(page_segments))

    md_lines: list[str] = list(header_lines)
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    citations: dict[str, dict] = {}
    text_id = 0
    for page_idx, segment in enumerate(page_segments):
        for para in _split_paragraphs(segment):
            md_lines.append(f"[TEXT:{text_id}]")
            md_lines.append("")
            md_lines.append(para)
            md_lines.append("")
            citations[f"#/texts/{text_id}"] = {
                "page": page_idx + 1,
                "line_start": None,
                "line_end": None,
                "bates": page_bates[page_idx],
                "column": None,
                "paragraph_number": None,
                "type": "page_only",
            }
            text_id += 1

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    cite_path.write_text(json.dumps(citations, indent=2), encoding="utf-8")
    return md_path, cite_path


def _convert_native(rec: DocRecord, production_root: Path, converted_dir: Path) -> Optional[str]:
    """Run format_handlers on a native file; return extracted body text or None."""
    from lit_pipeline.format_handlers import FormatHandler

    src = _resolve_rel(rec.native_path, production_root)
    if src is None:
        return None
    ext = (rec.doc_ext or src.suffix.lstrip(".")).lower()
    if ext not in PARSEABLE_NATIVE_EXTS:
        return None

    # format_handlers writes a .md to its output dir; we want its text body so
    # we can re-wrap with our standard header + markers. Use a scratch dir.
    scratch = converted_dir / ".loadfile_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    # Copy the native to a temp path keyed by bates so format_handlers names
    # its output as {bates}.md (it uses input_path.stem).
    bates_lower = rec.beg_bates.lower()
    scratch_input = scratch / f"{bates_lower}.{ext}"
    scratch_input.write_bytes(src.read_bytes())

    fh = FormatHandler(str(scratch))
    result = fh.convert(scratch_input)
    if not result.md_path:
        # Conversion failed; clean up scratch input and bail.
        try:
            scratch_input.unlink()
        except OSError:
            pass
        return None

    md_path = Path(result.md_path)
    body = md_path.read_text(encoding="utf-8", errors="replace")
    # Clean up scratch artifacts.
    try:
        md_path.unlink()
        scratch_input.unlink()
    except OSError:
        pass
    return body


def _find_dat(production_root: Path) -> Path:
    """Locate the .dat for a production: in the production dir itself, a
    'load files' sibling/child dir, or the parent — matched by stem when
    several are present."""
    candidates: list[Path] = []
    for base in (production_root, production_root / "load files",
                 production_root.parent / "load files", production_root.parent):
        if base.is_dir():
            candidates.extend(sorted(base.glob("*.dat")))
    if not candidates:
        raise FileNotFoundError(f"No .dat file found for {production_root}")
    for c in candidates:
        if c.stem.lower() == production_root.name.lower():
            return c
    return candidates[0]


def ingest_production(
    production_root: Path,
    corpus_root: Path,
    meta_dir: Path,
    run_chunker: bool = True,
    dat_path: Optional[Path] = None,
    opt_path: Optional[Path] = None,
    paginate: bool = True,
    ocr_cache: Optional[Path] = None,
) -> dict:
    """Stage one production's documents into the corpus, then chunk.

    Multi-page documents whose vendor text lacks form feeds are paginated via
    page_align (vendor form feeds when they match the produced page count,
    Tesseract-anchor alignment against the .opt page images otherwise);
    documents that can't be paginated safely are ingested with document-level
    Bates cites and listed in flagged_pagination.json.

    Writes:
      - corpus_root/converted/{bates}.md
      - corpus_root/converted/{bates}_citations.json
      - corpus_root/converted/{bates}_chunks.json (if run_chunker=True)
      - meta_dir/manifest.json
      - meta_dir/flagged_no_content.json
      - meta_dir/flagged_pagination.json
      - meta_dir/ingest_stats.json
    """
    from lit_pipeline.page_align import parse_opt, paginate_vendor_text

    if dat_path is None:
        dat_path = _find_dat(production_root)

    if opt_path is None:
        sibling = dat_path.with_suffix(".opt")
        opt_path = sibling if sibling.exists() else None
    opt_map: dict[str, list[str]] = parse_opt(opt_path) if opt_path else {}
    if paginate and not opt_map:
        logger.warning(
            "No .opt image cross-reference found for %s — multi-page documents "
            "without vendor form feeds will carry document-level cites only",
            dat_path.name,
        )

    converted_dir = corpus_root / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if ocr_cache is None:
        ocr_cache = corpus_root / ".ocr_cache"

    records = parse_dat(dat_path)
    logger.info("Parsed %d records from %s", len(records), dat_path.name)

    chunker = None
    doc_type = None
    if run_chunker:
        from lit_pipeline.chunk_documents import DocumentChunker
        from lit_pipeline.citation_types import DocumentType
        chunker = DocumentChunker(str(converted_dir))
        doc_type = DocumentType.EXHIBIT

    try:
        from tqdm import tqdm
        progress = tqdm(records, desc=f"ingest {dat_path.stem}", unit="doc")
    except ImportError:
        progress = records

    manifest: list[dict] = []
    flagged: list[dict] = []
    flagged_pagination: list[dict] = []
    stats = {
        "production": production_root.name,
        "total_records": len(records),
        "text_route": 0,
        "native_route": 0,
        "skipped_no_content": 0,
        "pagination": {"single": 0, "vendor-ff": 0, "aligned": 0,
                       "unpaged-mismatch": 0, "unpaged-no-ocr": 0, "n/a": 0},
        "pagination_flagged": 0,
        "chunks_written": 0,
        "chunk_failures": 0,
    }

    for rec in progress:
        body_text: Optional[str] = None
        route: str = ""

        # Try TEXT route first
        if rec.text_path:
            text_file = _resolve_rel(rec.text_path, production_root)
            data = _read_text_bytes(text_file) if text_file else b""
            if _is_usable_text(data):
                body_text = data.decode("utf-8", errors="replace")
                route = "text"

        # Fall back to native
        if body_text is None and rec.native_path:
            body_text = _convert_native(rec, production_root, converted_dir)
            if body_text is not None:
                route = "native"

        if body_text is None:
            flagged.append({
                "bates": rec.beg_bates,
                "filename": rec.filename,
                "doc_ext": rec.doc_ext,
                "text_path": rec.text_path,
                "native_path": rec.native_path,
                "reason": "neither vendor text nor parseable native available",
            })
            stats["skipped_no_content"] += 1
            continue

        # Recover page boundaries so citations can carry per-page Bates.
        # Native-route text has no page images to align against; it stays
        # document-level.
        pagination_route = "n/a"
        pagination_flagged = False
        if paginate and route == "text":
            try:
                n_pages = int(rec.pages or "1")
            except ValueError:
                n_pages = 1
            tif_paths = None
            if opt_map.get(rec.beg_bates):
                resolved = [_resolve_rel(t, production_root)
                            for t in opt_map[rec.beg_bates]]
                if all(p is not None for p in resolved):
                    tif_paths = resolved
            result = paginate_vendor_text(
                body_text, n_pages, tif_paths=tif_paths, ocr_cache=ocr_cache,
            )
            body_text = result.text
            pagination_route = result.route
            pagination_flagged = result.flagged
            stats["pagination"][result.route] += 1
            if result.flagged:
                stats["pagination_flagged"] += 1
                flagged_pagination.append({
                    "bates": rec.beg_bates,
                    "bates_range": [rec.beg_bates, rec.end_bates],
                    "filename": rec.filename,
                    "pages": n_pages,
                    "route": result.route,
                    "reason": result.reason,
                })
        else:
            stats["pagination"]["n/a"] += 1

        md_path, _ = _write_md_and_citations(rec, body_text, converted_dir)

        if route == "text":
            stats["text_route"] += 1
        else:
            stats["native_route"] += 1

        # Chunk immediately so we have _chunks.json ready for the indexer.
        if chunker is not None:
            try:
                chunks = chunker.chunk_document(
                    stem=rec.beg_bates.lower(),
                    doc_type=doc_type,
                    source_file=rec.filename or rec.beg_bates,
                    source_path=str(md_path),
                )
                if chunks:
                    stats["chunks_written"] += len(chunks)
                else:
                    stats["chunk_failures"] += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("Chunker failed on %s: %s", rec.beg_bates, e)
                stats["chunk_failures"] += 1

        manifest.append({
            "bates": rec.beg_bates,
            "bates_range": [rec.beg_bates, rec.end_bates],
            "attach_range": [rec.beg_attach, rec.end_attach],
            "custodian": rec.custodian,
            "filename": rec.filename,
            "doc_ext": rec.doc_ext,
            "pages": rec.pages,
            "author": rec.author,
            "date_sent": rec.date_sent,
            "date_created": rec.date_created,
            "date_last_mod": rec.date_last_mod,
            "subject": rec.subject,
            "from": rec.from_field,
            "to": rec.to_field,
            "cc": rec.cc,
            "confidentiality": rec.confidentiality,
            "production": rec.prod_vol,
            "route": route,
            "pagination": pagination_route,
            "pagination_flagged": pagination_flagged,
        })

    # Clean up scratch dir if empty
    scratch = converted_dir / ".loadfile_scratch"
    if scratch.exists():
        try:
            scratch.rmdir()
        except OSError:
            pass

    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (meta_dir / "flagged_no_content.json").write_text(json.dumps(flagged, indent=2))
    (meta_dir / "flagged_pagination.json").write_text(json.dumps(flagged_pagination, indent=2))
    (meta_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2))

    return stats


def find_productions(root: Path) -> list[tuple[Path, Path]]:
    """Discover (dat_path, production_root) pairs under a delivery root.

    Looks for .dat files in root, root/'load files', and one level of
    subdirectories; each dat is matched to the production dir named after its
    stem when one exists, else the root itself.
    """
    dats: list[Path] = []
    for base in [root, root / "load files"] + [d for d in root.iterdir() if d.is_dir()]:
        if base.is_dir():
            dats.extend(sorted(base.glob("*.dat")))
    # de-dup by production stem (vendors often ship a copy of the load file
    # inside the production folder as well); first occurrence wins, and the
    # search order above prefers the delivery root / 'load files' copy
    seen: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    for dat in dats:
        key = dat.stem.lower()
        if key in seen:
            continue
        seen.add(key)
        prod_dir = root / dat.stem
        pairs.append((dat, prod_dir if prod_dir.is_dir() else root))
    return pairs


def ingest_tree(
    root: Path,
    corpus_root: Path,
    meta_root: Path,
    run_chunker: bool = True,
    paginate: bool = True,
    ocr_cache: Optional[Path] = None,
) -> list[dict]:
    """Ingest every load-file production found under a delivery root."""
    pairs = find_productions(root)
    if not pairs:
        raise FileNotFoundError(f"No .dat load files found under {root}")
    all_stats = []
    for dat, prod_dir in pairs:
        logger.info("Ingesting production %s (root %s)", dat.stem, prod_dir)
        all_stats.append(ingest_production(
            prod_dir, corpus_root, meta_root / dat.stem,
            run_chunker=run_chunker, dat_path=dat, paginate=paginate,
            ocr_cache=ocr_cache,
        ))
    return all_stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ingest Concordance-style production(s) into a lit-doc-pipeline corpus."
    )
    p.add_argument("production_root", help="Extracted production dir, or a delivery root holding several productions plus their load files")
    p.add_argument("corpus_root", help="Existing lit-doc-pipeline corpus root (will write to its converted/)")
    p.add_argument("meta_dir", help="Metadata output root (manifest + flagged lists + stats, per production)")
    p.add_argument("--dat", help="Explicit .dat path (single production)")
    p.add_argument("--opt", help="Explicit .opt path (single production)")
    p.add_argument("--no-chunk", action="store_true", help="Skip chunking step (only write .md + _citations.json)")
    p.add_argument("--no-paginate", action="store_true", help="Skip page-boundary recovery (document-level cites)")
    p.add_argument("--ocr-cache", help="Directory for cached per-page OCR (default: <corpus_root>/.ocr_cache)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.production_root).resolve()
    ocr_cache = Path(args.ocr_cache).resolve() if args.ocr_cache else None
    if args.dat:
        stats = ingest_production(
            root,
            Path(args.corpus_root).resolve(),
            Path(args.meta_dir).resolve(),
            run_chunker=not args.no_chunk,
            dat_path=Path(args.dat).resolve(),
            opt_path=Path(args.opt).resolve() if args.opt else None,
            paginate=not args.no_paginate,
            ocr_cache=ocr_cache,
        )
        print(json.dumps(stats, indent=2))
    else:
        all_stats = ingest_tree(
            root,
            Path(args.corpus_root).resolve(),
            Path(args.meta_dir).resolve(),
            run_chunker=not args.no_chunk,
            paginate=not args.no_paginate,
            ocr_cache=ocr_cache,
        )
        print(json.dumps(all_stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
