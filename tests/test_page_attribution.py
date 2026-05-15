"""
Tests for per-line page attribution (page_map / bates_map).

Guards against the class of bugs where chunk-level page ranges are preserved
but the per-line provenance needed for precise search attribution is lost.

Covers:
  - _build_line_maps: correct page/bates per line, forward-fill of gaps
  - _create_chunk: page_map/bates_map stored in citation dict
  - _chunk_generic / _chunk_expert_report / _chunk_deposition: maps produced
  - _find_query_page / _char_pos_to_line_idx: search-time page resolution
  - Serialization round-trip (to_dict → JSON → reload)
  - Backward compatibility with chunks that lack page_map
"""

import json
import tempfile
import shutil
from pathlib import Path

import pytest

from lit_pipeline.chunk_documents import DocumentChunker, ChunkMetadata
from lit_pipeline.citation_types import Chunk, DocumentType
from lit_pipeline.lit_doc_retriever import _find_query_page, _char_pos_to_line_idx


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def tmp_converted(tmp_path):
    """Provide a temp directory acting as converted_dir."""
    return tmp_path


@pytest.fixture
def chunker(tmp_converted):
    return DocumentChunker(str(tmp_converted))


def _write_md_and_citations(tmp_dir, stem, md_content, citations):
    """Helper: write a .md and _citations.json into tmp_dir."""
    (tmp_dir / f"{stem}.md").write_text(md_content)
    (tmp_dir / f"{stem}_citations.json").write_text(json.dumps(citations))


# ── _build_line_maps ────────────────────────────────────────────────


class TestBuildLineMaps:
    """Unit tests for the _build_line_maps helper."""

    def test_basic_mapping(self, chunker):
        """Each line with a text_id resolves to the correct page and bates."""
        entries = [
            ("line on page 5", "10"),
            ("line on page 6", "20"),
            ("line on page 7", "30"),
        ]
        citations = {
            "#/texts/10": {"page": 5, "bates": "B001"},
            "#/texts/20": {"page": 6, "bates": "B002"},
            "#/texts/30": {"page": 7, "bates": "B003"},
        }
        page_map, bates_map = chunker._build_line_maps(entries, citations)

        assert page_map == [5, 6, 7]
        assert bates_map == ["B001", "B002", "B003"]

    def test_forward_fill_gaps(self, chunker):
        """Lines without a text_id inherit page/bates from the preceding line."""
        entries = [
            ("first", "10"),
            ("gap line", None),
            ("another gap", None),
            ("back on track", "20"),
        ]
        citations = {
            "#/texts/10": {"page": 3, "bates": "X100"},
            "#/texts/20": {"page": 4, "bates": "X200"},
        }
        page_map, bates_map = chunker._build_line_maps(entries, citations)

        assert page_map == [3, 3, 3, 4]
        assert bates_map == ["X100", "X100", "X100", "X200"]

    def test_leading_none_stays_none(self, chunker):
        """Lines before any known text_id remain None (no backward fill)."""
        entries = [
            ("orphan line", None),
            ("known", "5"),
            ("another gap", None),
        ]
        citations = {
            "#/texts/5": {"page": 10, "bates": "Z1"},
        }
        page_map, bates_map = chunker._build_line_maps(entries, citations)

        assert page_map[0] is None
        assert page_map[1] == 10
        assert page_map[2] == 10
        assert bates_map[0] is None

    def test_empty_entries(self, chunker):
        """Empty entry list produces empty maps."""
        page_map, bates_map = chunker._build_line_maps([], {})
        assert page_map == []
        assert bates_map == []

    def test_missing_citation_key(self, chunker):
        """A text_id with no matching citation produces None (then forward-filled)."""
        entries = [
            ("known", "1"),
            ("missing citation", "999"),
            ("still missing", None),
        ]
        citations = {
            "#/texts/1": {"page": 2, "bates": "A"},
        }
        page_map, bates_map = chunker._build_line_maps(entries, citations)

        assert page_map == [2, 2, 2]  # 999 not found → None → forward-filled from 2
        assert bates_map == ["A", "A", "A"]

    def test_citation_without_bates(self, chunker):
        """Citations that have page but no bates produce None bates entries."""
        entries = [
            ("text", "1"),
        ]
        citations = {
            "#/texts/1": {"page": 7},  # no bates key
        }
        page_map, bates_map = chunker._build_line_maps(entries, citations)

        assert page_map == [7]
        assert bates_map == [None]

    def test_map_length_matches_entries(self, chunker):
        """page_map and bates_map must have exactly one entry per input line."""
        entries = [("l", str(i)) for i in range(20)]
        citations = {f"#/texts/{i}": {"page": i + 1} for i in range(20)}

        page_map, bates_map = chunker._build_line_maps(entries, citations)

        assert len(page_map) == 20
        assert len(bates_map) == 20


# ── _create_chunk storage ───────────────────────────────────────────


class TestCreateChunkMaps:
    """Verify _create_chunk stores / omits page_map and bates_map."""

    def test_maps_stored_in_citation(self, chunker):
        """When page_map is provided, it appears in the chunk citation dict."""
        metadata = ChunkMetadata(pages=[5, 6])
        chunk = chunker._create_chunk(
            "line1\nline2", metadata, "stem", "src.pdf", 0,
            DocumentType.UNKNOWN,
            page_map=[5, 6], bates_map=["B1", "B2"],
        )

        assert chunk.citation["page_map"] == [5, 6]
        assert chunk.citation["bates_map"] == ["B1", "B2"]

    def test_maps_omitted_when_none(self, chunker):
        """When page_map is not provided, citation dict has no page_map key."""
        metadata = ChunkMetadata(pages=[1])
        chunk = chunker._create_chunk(
            "text", metadata, "stem", "src.pdf", 0,
            DocumentType.UNKNOWN,
        )

        assert "page_map" not in chunk.citation
        assert "bates_map" not in chunk.citation

    def test_bates_map_omitted_when_all_none(self, chunker):
        """bates_map is omitted if every entry is None."""
        metadata = ChunkMetadata(pages=[1])
        chunk = chunker._create_chunk(
            "text", metadata, "stem", "src.pdf", 0,
            DocumentType.UNKNOWN,
            page_map=[1], bates_map=[None],
        )

        assert "page_map" in chunk.citation
        assert "bates_map" not in chunk.citation


# ── Generic chunking ────────────────────────────────────────────────


class TestGenericChunkingMaps:
    """Verify _chunk_generic produces page_map in every chunk."""

    def test_single_chunk_has_page_map(self, chunker, tmp_converted):
        """A small document that fits in one chunk still gets a page_map."""
        stem = "generic_small"
        md = "[TEXT:1]\nHello world\n[TEXT:2]\nSecond line"
        citations = {
            "#/texts/1": {"page": 1, "bates": "G001"},
            "#/texts/2": {"page": 1, "bates": "G001"},
        }
        _write_md_and_citations(tmp_converted, stem, md, citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")

        assert len(chunks) >= 1
        chunk = chunks[0]
        pm = chunk.citation.get("page_map", [])
        assert len(pm) > 0, "page_map should not be empty"
        assert len(pm) == len(chunk.core_text.split("\n"))

    def test_multi_page_chunk_has_distinct_pages(self, chunker, tmp_converted):
        """A chunk spanning pages 3 and 4 has both values in page_map."""
        stem = "generic_multi"
        # Build content with two text_ids on different pages
        lines = []
        lines.append("[TEXT:10]")
        lines.extend([f"Page three content line {i}" for i in range(5)])
        lines.append("[TEXT:20]")
        lines.extend([f"Page four content line {i}" for i in range(5)])

        citations = {
            "#/texts/10": {"page": 3, "bates": "M010"},
            "#/texts/20": {"page": 4, "bates": "M020"},
        }
        _write_md_and_citations(tmp_converted, stem, "\n".join(lines), citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")

        assert len(chunks) >= 1
        pm = chunks[0].citation.get("page_map", [])
        unique_pages = set(p for p in pm if p is not None)
        assert 3 in unique_pages
        assert 4 in unique_pages

    def test_page_map_length_equals_core_text_lines(self, chunker, tmp_converted):
        """page_map must have exactly as many entries as lines in core_text."""
        stem = "generic_len"
        lines = ["[TEXT:1]"] + [f"line {i}" for i in range(12)]
        citations = {"#/texts/1": {"page": 1}}
        _write_md_and_citations(tmp_converted, stem, "\n".join(lines), citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")

        for chunk in chunks:
            pm = chunk.citation.get("page_map", [])
            n_lines = len(chunk.core_text.split("\n"))
            assert len(pm) == n_lines, (
                f"page_map length {len(pm)} != core_text lines {n_lines} "
                f"in {chunk.chunk_id}"
            )


# ── Expert report chunking ──────────────────────────────────────────


class TestExpertReportChunkingMaps:
    """Verify _chunk_expert_report produces page_map in every chunk."""

    def test_expert_report_has_page_map(self, chunker, tmp_converted):
        """Expert report chunks carry per-line page_map."""
        stem = "expert_test"
        md_lines = ["[TEXT:1]", "1. First paragraph content here."]
        md_lines += ["[TEXT:2]", "Continuation of first paragraph."]
        citations = {
            "#/texts/1": {"page": 10, "paragraph_number": 1},
            "#/texts/2": {"page": 10, "paragraph_number": 1},
        }
        _write_md_and_citations(tmp_converted, stem, "\n".join(md_lines), citations)

        chunks = chunker.chunk_document(stem, DocumentType.EXPERT_REPORT, "report.pdf")

        assert len(chunks) >= 1
        pm = chunks[0].citation.get("page_map", [])
        assert len(pm) > 0
        assert all(p == 10 for p in pm if p is not None)

    def test_expert_report_map_length(self, chunker, tmp_converted):
        """page_map length must equal core_text line count for expert reports."""
        stem = "expert_len"
        md_lines = []
        for i in range(1, 6):
            md_lines.append(f"[TEXT:{i}]")
            md_lines.append(f"{i}. Paragraph {i} content goes here.")
        citations = {
            f"#/texts/{i}": {"page": i, "paragraph_number": i}
            for i in range(1, 6)
        }
        _write_md_and_citations(tmp_converted, stem, "\n".join(md_lines), citations)

        chunks = chunker.chunk_document(stem, DocumentType.EXPERT_REPORT, "report.pdf")

        for chunk in chunks:
            pm = chunk.citation.get("page_map", [])
            n_lines = len(chunk.core_text.split("\n"))
            assert len(pm) == n_lines, (
                f"page_map length {len(pm)} != core_text lines {n_lines} "
                f"in {chunk.chunk_id}"
            )


# ── Deposition chunking ─────────────────────────────────────────────


class TestDepositionChunkingMaps:
    """Verify _chunk_deposition produces page_map in every chunk."""

    def _make_deposition_md(self):
        """Create simple deposition markdown with Q/A pairs across 2 pages."""
        lines = [
            "[PAGE:5]",
            " 1  Q  What is your name?",
            " 2  A  My name is John Smith.",
            " 3  Q  Where do you work?",
            " 4  A  I work at Acme Corp.",
            "[PAGE:6]",
            " 1  Q  Describe the incident.",
            " 2  A  It happened on Tuesday.",
        ]
        return "\n".join(lines)

    def _make_deposition_citations(self):
        """Citations keyed by line_P{page}_L{line}."""
        return {
            "line_P5_L1": {"page": 5, "bates": "DEP_001", "type": "transcript_line"},
            "line_P5_L2": {"page": 5, "bates": "DEP_001", "type": "transcript_line"},
            "line_P5_L3": {"page": 5, "bates": "DEP_001", "type": "transcript_line"},
            "line_P5_L4": {"page": 5, "bates": "DEP_001", "type": "transcript_line"},
            "line_P6_L1": {"page": 6, "bates": "DEP_002", "type": "transcript_line"},
            "line_P6_L2": {"page": 6, "bates": "DEP_002", "type": "transcript_line"},
        }

    def test_deposition_has_page_map(self, chunker, tmp_converted):
        """Deposition chunks carry per-line page_map."""
        stem = "dep_test"
        _write_md_and_citations(
            tmp_converted, stem,
            self._make_deposition_md(),
            self._make_deposition_citations(),
        )

        chunks = chunker.chunk_document(stem, DocumentType.DEPOSITION, "dep.pdf")

        assert len(chunks) >= 1
        for chunk in chunks:
            pm = chunk.citation.get("page_map")
            assert pm is not None, f"Missing page_map on {chunk.chunk_id}"

    def test_deposition_page_map_pages_correct(self, chunker, tmp_converted):
        """Page values in page_map match the [PAGE:N] markers."""
        stem = "dep_pages"
        _write_md_and_citations(
            tmp_converted, stem,
            self._make_deposition_md(),
            self._make_deposition_citations(),
        )

        chunks = chunker.chunk_document(stem, DocumentType.DEPOSITION, "dep.pdf")

        all_pages = []
        for chunk in chunks:
            pm = chunk.citation.get("page_map", [])
            all_pages.extend(p for p in pm if p is not None)

        assert 5 in all_pages, "Should contain page 5"
        assert 6 in all_pages, "Should contain page 6"

    def test_deposition_bates_map_populated(self, chunker, tmp_converted):
        """Bates values in bates_map match citations."""
        stem = "dep_bates"
        _write_md_and_citations(
            tmp_converted, stem,
            self._make_deposition_md(),
            self._make_deposition_citations(),
        )

        chunks = chunker.chunk_document(stem, DocumentType.DEPOSITION, "dep.pdf")

        all_bates = []
        for chunk in chunks:
            bm = chunk.citation.get("bates_map", [])
            all_bates.extend(b for b in bm if b is not None)

        assert "DEP_001" in all_bates
        assert "DEP_002" in all_bates

    def test_deposition_map_length(self, chunker, tmp_converted):
        """page_map length must equal core_text line count for depositions."""
        stem = "dep_len"
        _write_md_and_citations(
            tmp_converted, stem,
            self._make_deposition_md(),
            self._make_deposition_citations(),
        )

        chunks = chunker.chunk_document(stem, DocumentType.DEPOSITION, "dep.pdf")

        for chunk in chunks:
            pm = chunk.citation.get("page_map", [])
            n_lines = len(chunk.core_text.split("\n"))
            assert len(pm) == n_lines, (
                f"page_map length {len(pm)} != core_text lines {n_lines} "
                f"in {chunk.chunk_id}"
            )


# ── Query-to-page resolution ────────────────────────────────────────


class TestCharPosToLineIdx:
    """Unit tests for _char_pos_to_line_idx."""

    def test_first_line(self):
        assert _char_pos_to_line_idx("abc\ndef\nghi", 0) == 0
        assert _char_pos_to_line_idx("abc\ndef\nghi", 2) == 0

    def test_second_line(self):
        assert _char_pos_to_line_idx("abc\ndef\nghi", 4) == 1
        assert _char_pos_to_line_idx("abc\ndef\nghi", 6) == 1

    def test_third_line(self):
        assert _char_pos_to_line_idx("abc\ndef\nghi", 8) == 2

    def test_single_line(self):
        assert _char_pos_to_line_idx("no newlines", 5) == 0

    def test_empty_string(self):
        assert _char_pos_to_line_idx("", 0) == 0


class TestFindQueryPage:
    """Unit tests for _find_query_page."""

    @staticmethod
    def _make_chunk(core_text, page_map, bates_map=None):
        """Build a Chunk with the given page_map."""
        citation = {"pdf_pages": sorted(set(page_map)), "page_map": page_map}
        if bates_map:
            citation["bates_map"] = bates_map
        return Chunk(
            chunk_id="test_chunk_0001",
            core_text=core_text,
            pages=sorted(set(page_map)),
            citation=citation,
            citation_string="Test, pp. 1-3",
        )

    def test_exact_match_first_line(self):
        chunk = self._make_chunk(
            "alpha bravo\ncharlie delta\necho foxtrot",
            [1, 2, 3],
            ["B1", "B2", "B3"],
        )
        page, bates = _find_query_page("alpha bravo", chunk)
        assert page == 1
        assert bates == "B1"

    def test_exact_match_middle_line(self):
        chunk = self._make_chunk(
            "alpha bravo\ncharlie delta\necho foxtrot",
            [1, 2, 3],
            ["B1", "B2", "B3"],
        )
        page, bates = _find_query_page("charlie delta", chunk)
        assert page == 2
        assert bates == "B2"

    def test_exact_match_last_line(self):
        chunk = self._make_chunk(
            "alpha bravo\ncharlie delta\necho foxtrot",
            [1, 2, 3],
        )
        page, bates = _find_query_page("echo foxtrot", chunk)
        assert page == 3
        assert bates is None  # no bates_map

    def test_case_insensitive_match(self):
        chunk = self._make_chunk("Hello World\nfoo bar", [5, 6])
        page, _ = _find_query_page("HELLO WORLD", chunk)
        assert page == 5

    def test_substring_match(self):
        """Query that's a substring of a line still resolves."""
        chunk = self._make_chunk(
            "The quick brown fox\njumps over the lazy dog",
            [10, 11],
        )
        page, _ = _find_query_page("brown fox", chunk)
        assert page == 10

    def test_fallback_term_overlap(self):
        """When no exact substring match, fall back to term overlap."""
        chunk = self._make_chunk(
            "wireless technology standards\npower management protocol\nbattery savings mode",
            [1, 2, 3],
        )
        # "power protocol" doesn't appear verbatim but overlaps with line 2
        page, _ = _find_query_page("power protocol", chunk)
        assert page == 2

    def test_no_page_map_returns_none(self):
        """Chunks without page_map gracefully return (None, None)."""
        chunk = Chunk(
            chunk_id="old_chunk",
            core_text="some text",
            pages=[1],
            citation={"pdf_pages": [1]},
            citation_string="Old, p. 1",
        )
        page, bates = _find_query_page("some text", chunk)
        assert page is None
        assert bates is None

    def test_empty_page_map_returns_none(self):
        """An empty page_map list is treated the same as missing."""
        chunk = Chunk(
            chunk_id="empty_map",
            core_text="text",
            pages=[1],
            citation={"pdf_pages": [1], "page_map": []},
            citation_string="X",
        )
        page, bates = _find_query_page("text", chunk)
        assert page is None
        assert bates is None

    def test_no_match_at_all(self):
        """Query with zero overlap returns (None, None)."""
        chunk = self._make_chunk("alpha bravo", [1])
        page, bates = _find_query_page("zzzzz qqqqq", chunk)
        assert page is None
        assert bates is None

    def test_multiline_query_resolves_to_start_line(self):
        """A query spanning lines resolves to the line where the match starts."""
        chunk = self._make_chunk(
            "start of sentence\ncontinuation here\nend of paragraph",
            [5, 6, 7],
        )
        # No verbatim match — falls back to overlap.
        # "start sentence end" overlaps with lines 0 and 2.
        page, _ = _find_query_page("start sentence end", chunk)
        assert page in (5, 7)  # either is acceptable


# ── Serialization round-trip ─────────────────────────────────────────


class TestSerializationRoundTrip:
    """Verify page_map/bates_map survive to_dict → JSON → reload."""

    def test_round_trip_preserves_maps(self):
        """page_map and bates_map survive JSON serialization."""
        chunk = Chunk(
            chunk_id="rt_chunk_0001",
            core_text="line1\nline2\nline3",
            pages=[5, 6],
            citation={
                "pdf_pages": [5, 6],
                "page_map": [5, 5, 6],
                "bates_map": ["B1", "B1", "B2"],
            },
            citation_string="Test, pp. 5-6",
            tokens=3,
        )
        d = chunk.to_dict()
        json_str = json.dumps(d)
        reloaded = json.loads(json_str)

        assert reloaded["citation"]["page_map"] == [5, 5, 6]
        assert reloaded["citation"]["bates_map"] == ["B1", "B1", "B2"]

    def test_round_trip_without_maps(self):
        """Chunks without maps serialize cleanly (no KeyError on reload)."""
        chunk = Chunk(
            chunk_id="rt_old_0001",
            core_text="old chunk",
            pages=[1],
            citation={"pdf_pages": [1], "bates_range": []},
            citation_string="Old, p. 1",
        )
        d = chunk.to_dict()
        json_str = json.dumps(d)
        reloaded = json.loads(json_str)

        assert "page_map" not in reloaded["citation"]
        assert "bates_map" not in reloaded["citation"]

    def test_chunk_file_round_trip(self, tmp_path):
        """Write chunks to a JSON file and read back — maps intact."""
        chunks = [
            Chunk(
                chunk_id="file_rt_0001",
                core_text="alpha\nbeta",
                pages=[1, 2],
                citation={
                    "pdf_pages": [1, 2],
                    "page_map": [1, 2],
                    "bates_map": ["A", "B"],
                    "bates_range": ["A", "B"],
                },
                citation_string="Test, pp. 1-2",
                tokens=2,
            ),
        ]
        out = tmp_path / "test_chunks.json"
        out.write_text(json.dumps([c.to_dict() for c in chunks]))

        loaded = json.loads(out.read_text())
        assert loaded[0]["citation"]["page_map"] == [1, 2]
        assert loaded[0]["citation"]["bates_map"] == ["A", "B"]


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
    """Ensure old chunks (no page_map) don't break any code paths."""

    def test_find_query_page_graceful_on_old_chunk(self):
        """_find_query_page returns (None, None) for pre-existing chunks."""
        chunk = Chunk(
            chunk_id="legacy_0001",
            core_text="This is old content without per-line maps.",
            pages=[3, 4, 5],
            citation={
                "pdf_pages": [3, 4, 5],
                "bates_range": ["LEGACY_001", "LEGACY_002"],
            },
            citation_string="Legacy Doc, pp. 3-5",
        )
        page, bates = _find_query_page("old content", chunk)
        assert page is None
        assert bates is None

    def test_old_chunk_still_has_pdf_pages(self):
        """Chunk-level pdf_pages field is always present regardless of page_map."""
        chunk = Chunk(
            chunk_id="compat_0001",
            core_text="text",
            pages=[1, 2],
            citation={"pdf_pages": [1, 2], "bates_range": []},
            citation_string="X",
        )
        assert chunk.citation["pdf_pages"] == [1, 2]
        assert "page_map" not in chunk.citation


# ── Integration: page_map resolves to correct page for search ────────


class TestSearchPageAttribution:
    """
    Integration-level tests: build chunks from markdown + citations,
    then verify _find_query_page pins the query to the right page.
    """

    def test_query_on_page_3_of_3_page_chunk(self, chunker, tmp_converted):
        """Text unique to page 7 is attributed to page 7, not 5 or 6."""
        stem = "attr_test"
        md_lines = [
            "[TEXT:100]",
            "Content on page five alpha.",
            "[TEXT:101]",
            "Content on page six bravo.",
            "[TEXT:102]",
            "Unique phrase on page seven charlie.",
        ]
        citations = {
            "#/texts/100": {"page": 5, "bates": "AT_005"},
            "#/texts/101": {"page": 6, "bates": "AT_006"},
            "#/texts/102": {"page": 7, "bates": "AT_007"},
        }
        _write_md_and_citations(tmp_converted, stem, "\n".join(md_lines), citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")

        # Find the chunk containing our target text
        target_chunk = None
        for c in chunks:
            if "Unique phrase on page seven" in c.core_text:
                target_chunk = c
                break
        assert target_chunk is not None, "Target chunk not found"

        page, bates = _find_query_page("Unique phrase on page seven", target_chunk)
        assert page == 7
        assert bates == "AT_007"

    def test_query_resolves_first_page_not_last(self, chunker, tmp_converted):
        """Text on page 5 should not be attributed to page 6."""
        stem = "attr_first"
        md_lines = [
            "[TEXT:10]",
            "First page exclusive content.",
            "[TEXT:20]",
            "Second page different content.",
        ]
        citations = {
            "#/texts/10": {"page": 5, "bates": "FIRST_005"},
            "#/texts/20": {"page": 6, "bates": "FIRST_006"},
        }
        _write_md_and_citations(tmp_converted, stem, "\n".join(md_lines), citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")
        chunk = chunks[0]

        page, bates = _find_query_page("First page exclusive", chunk)
        assert page == 5, f"Expected page 5, got {page}"
        assert bates == "FIRST_005"

    def test_deposition_query_resolves_correct_transcript_page(
        self, chunker, tmp_converted
    ):
        """In a deposition, a Q/A on page 6 should resolve to page 6."""
        stem = "dep_attr"
        md = "\n".join([
            "[PAGE:5]",
            " 1  Q  What is your role?",
            " 2  A  I am the lead engineer.",
            "[PAGE:6]",
            " 1  Q  Describe the TWT implementation.",
            " 2  A  TWT uses target wake time scheduling.",
        ])
        citations = {
            "line_P5_L1": {"page": 5, "bates": "D_005"},
            "line_P5_L2": {"page": 5, "bates": "D_005"},
            "line_P6_L1": {"page": 6, "bates": "D_006"},
            "line_P6_L2": {"page": 6, "bates": "D_006"},
        }
        _write_md_and_citations(tmp_converted, stem, md, citations)

        chunks = chunker.chunk_document(stem, DocumentType.DEPOSITION, "dep.pdf")
        assert len(chunks) >= 1

        # Find chunk containing the TWT text
        for c in chunks:
            if "TWT" in c.core_text:
                page, bates = _find_query_page("TWT implementation", c)
                assert page == 6, f"Expected page 6, got {page}"
                assert bates == "D_006"
                break
        else:
            pytest.fail("No chunk contained 'TWT'")

    def test_expert_report_query_resolves_to_paragraph_page(
        self, chunker, tmp_converted
    ):
        """In an expert report, a paragraph on page 12 resolves to page 12."""
        stem = "expert_attr"
        md = "\n".join([
            "[TEXT:50]",
            "1. The prior art fails to disclose the claimed invention.",
            "[TEXT:51]",
            "2. Dr. Smith's analysis is flawed because the test methodology was incorrect.",
        ])
        citations = {
            "#/texts/50": {"page": 11, "paragraph_number": 1},
            "#/texts/51": {"page": 12, "paragraph_number": 2},
        }
        _write_md_and_citations(tmp_converted, stem, md, citations)

        chunks = chunker.chunk_document(stem, DocumentType.EXPERT_REPORT, "report.pdf")
        assert len(chunks) >= 1

        for c in chunks:
            if "flawed" in c.core_text:
                page, _ = _find_query_page("test methodology was incorrect", c)
                assert page == 12, f"Expected page 12, got {page}"
                break
        else:
            pytest.fail("No chunk contained 'flawed'")
