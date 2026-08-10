"""Tests for lit_pipeline.page_align — page-boundary recovery for load-file
productions — and the loadfile_ingest routing that consumes it."""

import json
from pathlib import Path

import pytest

from lit_pipeline.page_align import (
    AlignmentStats,
    align_pages,
    paginate_vendor_text,
    parse_opt,
)
from lit_pipeline.loadfile_ingest import (
    HEADER_ALIASES,
    find_productions,
    ingest_production,
    parse_dat,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _page(text: str, salt: str) -> str:
    """Build a distinctive fake page: several lines with unique-ish tokens."""
    lines = [f"{salt} heading line for page", text]
    for i in range(6):
        lines.append(f"{salt}token{i} filler content number {salt}{i} more words")
    return "\n".join(lines)


def _noisy(text: str) -> str:
    """Simulate OCR noise: drop ~1 of every 7 words."""
    words = text.split()
    return " ".join(w for i, w in enumerate(words) if i % 7 != 3)


# ── align_pages (pure function) ──────────────────────────────────────────

class TestAlignPages:
    def test_three_page_alignment(self):
        pages = [_page("alpha bravo unique", "aaa"),
                 _page("charlie delta content", "bbb"),
                 _page("echo foxtrot words", "ccc")]
        vendor = "\n".join(pages)
        ocr = [_noisy(p) for p in pages]
        boundaries, stats = align_pages(vendor, ocr)
        assert len(boundaries) == 2
        segs = []
        prev = 0
        for b in boundaries:
            segs.append(vendor[prev:b]); prev = b
        segs.append(vendor[prev:])
        assert "aaa heading" in segs[0] and "bbbtoken0" not in segs[0]
        assert "bbb heading" in segs[1]
        assert "ccc heading" in segs[2]
        assert stats.pages_with_anchors == 3
        assert not stats.flagged

    def test_single_page_no_boundaries(self):
        boundaries, stats = align_pages("some text", ["some text"])
        assert boundaries == []
        assert stats.n_pages == 1

    def test_garbled_middle_page_does_not_cascade(self):
        """A page whose OCR shares nothing with the text must not derail the
        pages after it (the v1 greedy-cursor failure mode)."""
        pages = [_page("alpha bravo", "aaa"),
                 _page("charlie delta", "bbb"),
                 _page("echo foxtrot", "ccc"),
                 _page("golf hotel", "ddd")]
        vendor = "\n".join(pages)
        ocr = [_noisy(pages[0]), "@@@@ ???? ####", _noisy(pages[2]), _noisy(pages[3])]
        boundaries, stats = align_pages(vendor, ocr)
        assert len(boundaries) == 3
        # pages after the garbled one must still be found (no cascade):
        # the last segment contains page 4's content and none of page 3's heading
        last = vendor[boundaries[2]:]
        assert "ddd heading" in last
        assert "ccc heading" not in last
        assert stats.anchorless_pages == [2]

    def test_boundaries_monotonic(self):
        pages = [_page("w1 w2", f"p{i}") for i in range(6)]
        vendor = "\n".join(pages)
        boundaries, _ = align_pages(vendor, [_noisy(p) for p in pages])
        assert boundaries == sorted(boundaries)

    def test_flagged_when_mostly_anchorless(self):
        vendor = "word " * 500
        ocr = ["@@@" for _ in range(5)]
        _, stats = align_pages(vendor, ocr)
        assert stats.flagged


# ── paginate_vendor_text routing ─────────────────────────────────────────

class TestPaginateRouting:
    def test_single_page_route(self):
        r = paginate_vendor_text("hello", 1)
        assert r.route == "single" and not r.flagged

    def test_vendor_ff_exact(self):
        r = paginate_vendor_text("page one\fpage two\fpage three", 3)
        assert r.route == "vendor-ff" and not r.flagged
        assert r.text.count("\f") == 2

    def test_vendor_ff_trailing_formfeed_tolerated(self):
        r = paginate_vendor_text("page one\fpage two\f \n", 2)
        assert r.route == "vendor-ff"
        assert r.text.count("\f") == 1

    def test_mismatch_is_flagged_and_unpaged(self):
        # text paginated differently than the produced image (e.g. 4-up W2s)
        r = paginate_vendor_text("a\fb\fc\fd\fe", 2)
        assert r.route == "unpaged-mismatch" and r.flagged
        assert "\f" not in r.text  # never emit misleading page breaks

    def test_no_images_is_flagged(self):
        r = paginate_vendor_text("one two three", 3, tif_paths=None)
        assert r.route == "unpaged-no-ocr" and r.flagged

    def test_aligned_route_with_injected_ocr(self, tmp_path):
        pages = [_page("alpha bravo", "aaa"), _page("charlie delta", "bbb")]
        vendor = "\n".join(pages)
        fake_tifs = [tmp_path / "p1.tif", tmp_path / "p2.tif"]
        for p in fake_tifs:
            p.write_bytes(b"")
        ocr_by_name = {"p1.tif": _noisy(pages[0]), "p2.tif": _noisy(pages[1])}
        r = paginate_vendor_text(
            vendor, 2, tif_paths=fake_tifs,
            ocr_fn=lambda p: ocr_by_name[p.name],
        )
        assert r.route == "aligned" and not r.flagged
        segs = r.text.split("\f")
        assert len(segs) == 2
        assert "bbb heading" in segs[1]


# ── parse_opt ────────────────────────────────────────────────────────────

class TestParseOpt:
    def test_doc_grouping(self, tmp_path):
        opt = tmp_path / "x.opt"
        opt.write_text(
            "iLink_000001,VOL1,.\\P\\IMAGES\\iLink_000001.tif,Y,,,2\n"
            "iLink_000002,VOL1,.\\P\\IMAGES\\iLink_000002.tif,,,,\n"
            "iLink_000003,VOL1,.\\P\\IMAGES\\iLink_000003.tif,Y,,,1\n"
        )
        docs = parse_opt(opt)
        assert list(docs) == ["iLink_000001", "iLink_000003"]
        assert len(docs["iLink_000001"]) == 2
        assert len(docs["iLink_000003"]) == 1


# ── DAT header aliases ───────────────────────────────────────────────────

DELIM = "\x14"
Q = "\xfe"


def _dat_row(values: list[str]) -> str:
    return DELIM.join(f"{Q}{v}{Q}" for v in values)


class TestHeaderAliases:
    def test_iconect_style_headers(self, tmp_path):
        """BEGDOC/ENDDOC/PGCOUNT/TEXTFILE vocabulary (as in the iLink DATs)."""
        hdr = ["BEGDOC", "ENDDOC", "PGCOUNT", "CUSTODIAN", "FILENAME",
               "TEXTFILE", "NATIVEFILE", "RECORD TYPE"]
        row = ["iLink_000001", "iLink_000002", "2", "Smith, Al",
               "note.pdf", ".\\P\\TEXT\\iLink_000001.txt", "", "Edoc"]
        dat = tmp_path / "PRODX.dat"
        dat.write_text(_dat_row(hdr) + "\n" + _dat_row(row), encoding="utf-8")
        recs = parse_dat(dat)
        assert len(recs) == 1
        r = recs[0]
        assert r.beg_bates == "iLink_000001"
        assert r.end_bates == "iLink_000002"
        assert r.pages == "2"
        assert r.custodian == "Smith, Al"
        assert r.text_path.endswith("iLink_000001.txt")
        assert r.file_desc == "Edoc"

    def test_prodbeg_style_headers(self, tmp_path):
        hdr = ["PRODBEG", "PRODEND", "PAGES", "TEXTPATH"]
        row = ["AB0001", "AB0003", "3", "TEXT\\AB0001.txt"]
        dat = tmp_path / "PRODY.dat"
        dat.write_text(_dat_row(hdr) + "\n" + _dat_row(row), encoding="utf-8")
        r = parse_dat(dat)[0]
        assert r.beg_bates == "AB0001" and r.pages == "3"
        assert r.text_path.endswith("AB0001.txt")

    def test_alias_table_is_self_consistent(self):
        for canonical, aliases in HEADER_ALIASES.items():
            assert canonical in aliases


# ── end-to-end synthetic production ──────────────────────────────────────

def _build_production(root: Path) -> Path:
    """Delivery root: PRODX/ with TEXT + IMAGES, load files in 'load files'."""
    prod = root / "PRODX"
    (prod / "TEXT").mkdir(parents=True)
    (prod / "IMAGES").mkdir()
    lf = root / "load files"
    lf.mkdir()

    # doc 1: two pages, vendor form feed present
    (prod / "TEXT" / "AA0001.txt").write_text(
        "first page body text here\fsecond page body text here")
    # doc 2: single page
    (prod / "TEXT" / "AA0003.txt").write_text("lone page of doc two")
    # doc 3: three pages but NO form feeds and no OCR available -> flagged
    (prod / "TEXT" / "AA0004.txt").write_text(
        "three pages worth of text with no page breaks at all")

    hdr = ["BEGDOC", "ENDDOC", "PGCOUNT", "CUSTODIAN", "FILENAME",
           "TEXTFILE", "NATIVEFILE", "RECORD TYPE"]
    rows = [
        ["AA0001", "AA0002", "2", "C1", "a.pdf", ".\\PRODX\\TEXT\\AA0001.txt", "", "Edoc"],
        ["AA0003", "AA0003", "1", "C1", "b.pdf", ".\\PRODX\\TEXT\\AA0003.txt", "", "Edoc"],
        ["AA0004", "AA0006", "3", "C2", "c.pdf", ".\\PRODX\\TEXT\\AA0004.txt", "", "Edoc"],
    ]
    (lf / "PRODX.dat").write_text(
        _dat_row(hdr) + "\n" + "\n".join(_dat_row(r) for r in rows),
        encoding="utf-8")
    # opt intentionally references images that don't exist on disk for AA0004,
    # so pagination falls back to unpaged-no-ocr
    (lf / "PRODX.opt").write_text(
        "AA0001,PRODX,.\\PRODX\\IMAGES\\AA0001.tif,Y,,,2\n"
        "AA0002,PRODX,.\\PRODX\\IMAGES\\AA0002.tif,,,,\n"
        "AA0003,PRODX,.\\PRODX\\IMAGES\\AA0003.tif,Y,,,1\n"
        "AA0004,PRODX,.\\PRODX\\IMAGES\\AA0004.tif,Y,,,3\n"
        "AA0005,PRODX,.\\PRODX\\IMAGES\\AA0005.tif,,,,\n"
        "AA0006,PRODX,.\\PRODX\\IMAGES\\AA0006.tif,,,,\n")
    return prod


class TestIngestProduction:
    def test_find_productions(self, tmp_path):
        _build_production(tmp_path)
        pairs = find_productions(tmp_path)
        assert len(pairs) == 1
        dat, prod_dir = pairs[0]
        assert dat.name == "PRODX.dat"
        assert prod_dir.name == "PRODX"

    def test_ingest_routes_and_citations(self, tmp_path):
        _build_production(tmp_path)
        corpus = tmp_path / "corpus"
        meta = tmp_path / "meta"
        stats = ingest_production(
            tmp_path / "PRODX", corpus, meta, run_chunker=False,
            dat_path=tmp_path / "load files" / "PRODX.dat",
        )
        assert stats["text_route"] == 3
        assert stats["pagination"]["vendor-ff"] == 1
        assert stats["pagination"]["single"] == 1
        assert stats["pagination"]["unpaged-no-ocr"] == 1
        assert stats["pagination_flagged"] == 1

        # two-page doc: page 2 paragraph carries its own bates stamp
        cites = json.loads((corpus / "converted" / "aa0001_citations.json").read_text())
        pages = {c["page"] for c in cites.values()}
        assert pages == {1, 2}
        p2 = [c for c in cites.values() if c["page"] == 2]
        assert p2 and all(c["bates"] == "AA0002" for c in p2)

        # flagged doc: single page attribution, doc-level bates
        cites4 = json.loads((corpus / "converted" / "aa0004_citations.json").read_text())
        assert {c["page"] for c in cites4.values()} == {1}
        assert all(c["bates"] == "AA0004" for c in cites4.values())

        flagged = json.loads((meta / "flagged_pagination.json").read_text())
        assert [f["bates"] for f in flagged] == ["AA0004"]

    def test_manifest_carries_pagination(self, tmp_path):
        _build_production(tmp_path)
        corpus = tmp_path / "corpus"
        meta = tmp_path / "meta"
        ingest_production(
            tmp_path / "PRODX", corpus, meta, run_chunker=False,
            dat_path=tmp_path / "load files" / "PRODX.dat",
        )
        manifest = json.loads((meta / "manifest.json").read_text())
        by_bates = {m["bates"]: m for m in manifest}
        assert by_bates["AA0001"]["pagination"] == "vendor-ff"
        assert by_bates["AA0004"]["pagination"] == "unpaged-no-ocr"
        assert by_bates["AA0004"]["pagination_flagged"] is True
