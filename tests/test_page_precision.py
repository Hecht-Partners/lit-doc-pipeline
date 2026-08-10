"""
Tests for precise page-range formatting, per-line page span compression,
inline [PAGE:N] marker insertion, and filename-derived per-page Bates
stamping.

Covers:
  - format_page_ranges: gap-preserving page-range formatting
  - compute_page_spans: per-line page_map -> contiguous line-range spans
  - DocumentChunker._insert_page_markers: inline [PAGE:N] / [PAGE:N | BATES]
    marker insertion, alignment invariant, defensive length-mismatch guard
  - DocumentChunker.derive_bates_by_page: filename-encoded Bates range ->
    per-page Bates stamps
  - End-to-end via DocumentChunker.chunk_document
"""

import json

import pytest

from lit_pipeline.chunk_documents import (
    DocumentChunker,
    ChunkMetadata,
    format_page_ranges,
    compute_page_spans,
)
from lit_pipeline.citation_types import DocumentType


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


# ── format_page_ranges ───────────────────────────────────────────────


class TestFormatPageRanges:

    def test_gap_preservation(self):
        """Non-consecutive pages stay as separate runs, not one min-max span."""
        assert format_page_ranges([1, 6, 8, 9, 10]) == "pp. 1, 6, 8-10"

    def test_single_page(self):
        """A single page uses the singular 'p.' label."""
        assert format_page_ranges([4]) == "p. 4"

    def test_empty_input(self):
        """An empty collection formats as an empty string."""
        assert format_page_ranges([]) == ""

    def test_none_only_input(self):
        """A collection containing only None formats as an empty string."""
        assert format_page_ranges([None, None]) == ""

    def test_none_filtered_from_mixed_input(self):
        """None entries are dropped; remaining pages still compress to runs."""
        assert format_page_ranges([1, None, 2, None, 5]) == "pp. 1-2, 5"

    def test_unsorted_and_duplicate_input(self):
        """Input need not be sorted or deduplicated beforehand."""
        assert format_page_ranges([10, 1, 2, 1, 10]) == "pp. 1-2, 10"


# ── compute_page_spans ───────────────────────────────────────────────


class TestComputePageSpans:

    def test_single_contiguous_run(self):
        """A single page repeated across lines collapses to one span."""
        assert compute_page_spans([1, 1, 1]) == [
            {"page": 1, "line_start": 1, "line_end": 3},
        ]

    def test_none_gaps_omitted(self):
        """Lines with unknown page (None) produce no span entry."""
        assert compute_page_spans([1, None, 2]) == [
            {"page": 1, "line_start": 1, "line_end": 1},
            {"page": 2, "line_start": 3, "line_end": 3},
        ]

    def test_non_contiguous_same_page_kept_separate(self):
        """Returning to a previously-seen page after leaving it starts a
        new span rather than merging with the earlier one."""
        assert compute_page_spans([1, 1, 6, 6, 1]) == [
            {"page": 1, "line_start": 1, "line_end": 2},
            {"page": 6, "line_start": 3, "line_end": 4},
            {"page": 1, "line_start": 5, "line_end": 5},
        ]

    def test_one_indexed_line_numbers(self):
        """line_start/line_end are 1-indexed into core_text, not 0-indexed."""
        assert compute_page_spans([None, 5]) == [
            {"page": 5, "line_start": 2, "line_end": 2},
        ]

    def test_empty_page_map(self):
        """An empty page_map produces no spans."""
        assert compute_page_spans([]) == []


# ── DocumentChunker._insert_page_markers ─────────────────────────────


class TestInsertPageMarkers:

    def test_marker_before_first_known_line(self, chunker):
        """A marker is inserted before the very first line once its page
        is known, even though there was no prior page to transition from."""
        text, page_map, bates_map = chunker._insert_page_markers(
            "line1\nline2", [5, 6], None
        )
        assert text == "[PAGE:5]\nline1\n[PAGE:6]\nline2"
        assert page_map == [5, 5, 6, 6]
        assert bates_map is None

    def test_bates_variant_marker_format(self, chunker):
        """When bates is known for a transition line, the marker embeds it
        as '[PAGE:N | BATES]', and bates_map grows in lockstep with page_map."""
        text, page_map, bates_map = chunker._insert_page_markers(
            "line1\nline2", [5, 6], ["B1", "B2"]
        )
        assert text == "[PAGE:5 | B1]\nline1\n[PAGE:6 | B2]\nline2"
        assert page_map == [5, 5, 6, 6]
        assert bates_map == ["B1", "B1", "B2", "B2"]

    def test_leading_none_pages_get_no_marker_until_known(self, chunker):
        """Lines whose page is still unknown (None) get no marker; the
        first marker appears only once a concrete page is reached."""
        text, page_map, bates_map = chunker._insert_page_markers(
            "a\nb\nc", [None, None, 3], None
        )
        assert text == "a\nb\n[PAGE:3]\nc"
        assert page_map == [None, None, 3, 3]

    def test_length_mismatch_returns_input_unchanged(self, chunker):
        """If page_map doesn't align with the text's line count, the input
        is returned verbatim rather than risking a misaligned insertion."""
        text, page_map, bates_map = chunker._insert_page_markers(
            "a\nb\nc", [1, 2], None
        )
        assert text == "a\nb\nc"
        assert page_map == [1, 2]
        assert bates_map is None

    def test_alignment_invariant_after_insertion(self, chunker):
        """After insertion, page_map/bates_map length always equals the
        number of lines in the returned text, across multiple transitions."""
        text, page_map, bates_map = chunker._insert_page_markers(
            "l1\nl2\nl3\nl4",
            [1, 1, 2, 4],
            ["B1", "B1", "B2", "B4"],
        )
        assert len(page_map) == len(text.split("\n"))
        assert len(bates_map) == len(text.split("\n"))
        # Three transitions (1 -> 2 -> 4), each preceded by exactly one marker.
        assert text.split("\n").count(None) == 0  # sanity: no stray entries
        marker_lines = [l for l in text.split("\n") if l.startswith("[PAGE:")]
        assert marker_lines == ["[PAGE:1 | B1]", "[PAGE:2 | B2]", "[PAGE:4 | B4]"]

    def test_no_transitions_single_page(self, chunker):
        """A page_map with a single repeated page inserts exactly one
        leading marker and no others."""
        text, page_map, bates_map = chunker._insert_page_markers(
            "a\nb\nc", [7, 7, 7], None
        )
        assert text == "[PAGE:7]\na\nb\nc"
        assert page_map == [7, 7, 7, 7]


# ── DocumentChunker.derive_bates_by_page ──────────────────────────────


class TestDeriveBatesByPage:

    def test_happy_path_zero_padding_preserved(self):
        """A filename-encoded Bates range spanning exactly the document's
        page count fills each citation's bates, preserving zero-padding."""
        citations = {f"#/texts/{i}": {"page": i} for i in range(1, 13)}
        updated = DocumentChunker.derive_bates_by_page(
            "ABBOTT0000001-0000012 title.pdf", citations
        )
        assert updated == 12
        assert citations["#/texts/1"]["bates"] == "ABBOTT0000001"
        assert citations["#/texts/5"]["bates"] == "ABBOTT0000005"
        assert citations["#/texts/12"]["bates"] == "ABBOTT0000012"

    def test_guard_failure_on_page_count_mismatch(self):
        """When the filename range doesn't exactly match the page count,
        nothing is filled and the function returns 0."""
        citations = {f"#/texts/{i}": {"page": i} for i in range(1, 11)}  # 10 pages
        updated = DocumentChunker.derive_bates_by_page(
            "ABBOTT0000001-0000012 title.pdf", citations  # range implies 12
        )
        assert updated == 0
        assert "bates" not in citations["#/texts/1"]

    def test_noop_when_bates_already_present(self):
        """If any citation already carries a bates value, the filename
        range is never consulted and no citation is touched."""
        citations = {
            "#/texts/1": {"page": 1, "bates": "X1"},
            "#/texts/2": {"page": 2},
        }
        updated = DocumentChunker.derive_bates_by_page(
            "ABBOTT0000001-0000002 title.pdf", citations
        )
        assert updated == 0
        assert "bates" not in citations["#/texts/2"]

    def test_no_filename_match_is_noop(self):
        """A filename that doesn't start with a Bates range is a no-op."""
        citations = {"#/texts/1": {"page": 1}}
        updated = DocumentChunker.derive_bates_by_page("no_bates_here.pdf", citations)
        assert updated == 0
        assert "bates" not in citations["#/texts/1"]

    def test_prefix_uppercased_from_lowercase_filename(self):
        """A lowercase filename prefix is uppercased in the derived bates."""
        citations = {f"#/texts/{i}": {"page": i} for i in range(1, 13)}
        updated = DocumentChunker.derive_bates_by_page(
            "abbott0000001-0000012 title.pdf", citations
        )
        assert updated == 12
        assert citations["#/texts/5"]["bates"] == "ABBOTT0000005"

    def test_underscore_suffixed_stem_matches(self):
        """Regression: an underscore immediately after the end-Bates digits
        (the normalized-stem convention, e.g. 'abbott0000001_0000012_title')
        must still match — a trailing \\b fails at a digit->underscore
        boundary because underscore is a word character."""
        citations = {f"#/texts/{i}": {"page": i} for i in range(1, 13)}
        updated = DocumentChunker.derive_bates_by_page(
            "abbott0000001_0000012_250718_i3_ddd_attachment_d", citations
        )
        assert updated == 12
        assert citations["#/texts/1"]["bates"] == "ABBOTT0000001"
        assert citations["#/texts/12"]["bates"] == "ABBOTT0000012"

    def test_end_digits_not_split_by_partial_match(self):
        """The end-Bates number must be consumed whole: a range whose second
        number runs straight into more digits cannot match a prefix of it."""
        from lit_pipeline.chunk_documents import FILENAME_BATES_RE
        m = FILENAME_BATES_RE.match("abbott0000001_0000012_title")
        assert m is not None
        assert m.group(3) == "0000012"


# ── End-to-end via DocumentChunker.chunk_document ─────────────────────


class TestPagePrecisionEndToEnd:

    def test_generic_chunk_gets_page_markers_and_spans(self, chunker, tmp_converted):
        """A generic doc spanning non-adjacent pages 1 and 6 gets inline
        [PAGE:N] markers, a page_spans list consistent with page_map, and a
        citation_string that preserves the gap rather than collapsing it."""
        stem = "gap_pages"
        md = "\n".join([
            "[TEXT:1]",
            "Content on page one.",
            "[TEXT:2]",
            "Content on page six.",
        ])
        citations = {
            "#/texts/1": {"page": 1},
            "#/texts/2": {"page": 6},
        }
        _write_md_and_citations(tmp_converted, stem, md, citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")
        assert len(chunks) == 1
        chunk = chunks[0]

        assert "[PAGE:" in chunk.core_text
        assert "[PAGE:1]" in chunk.core_text
        assert "[PAGE:6]" in chunk.core_text

        page_map = chunk.citation["page_map"]
        assert len(page_map) == len(chunk.core_text.split("\n"))

        expected_spans = compute_page_spans(page_map)
        assert chunk.citation["page_spans"] == expected_spans
        assert {s["page"] for s in expected_spans} == {1, 6}

        assert chunk.citation_string.endswith("pp. 1, 6")
        assert "1-6" not in chunk.citation_string

    def test_deposition_gets_no_inserted_page_markers(self, chunker, tmp_converted):
        """Depositions are excluded from inline marker insertion (they cite
        transcript page:line instead), so core_text never contains [PAGE:."""
        stem = "dep_no_markers"
        md = "\n".join([
            "[PAGE:5]",
            " 1  Q  What is your name?",
            " 2  A  My name is John Smith.",
            "[PAGE:6]",
            " 1  Q  Where do you work?",
            " 2  A  I work at Acme Corp.",
        ])
        citations = {
            "line_P5_L1": {"page": 5, "bates": "DEP_001", "type": "transcript_line"},
            "line_P5_L2": {"page": 5, "bates": "DEP_001", "type": "transcript_line"},
            "line_P6_L1": {"page": 6, "bates": "DEP_002", "type": "transcript_line"},
            "line_P6_L2": {"page": 6, "bates": "DEP_002", "type": "transcript_line"},
        }
        _write_md_and_citations(tmp_converted, stem, md, citations)

        chunks = chunker.chunk_document(stem, DocumentType.DEPOSITION, "dep.pdf")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "[PAGE:" not in chunk.core_text

    def test_round_trip_preserves_page_spans(self, chunker, tmp_converted):
        """chunk.to_dict() -> json.dumps -> json.loads keeps page_spans intact."""
        stem = "roundtrip_spans"
        md = "\n".join([
            "[TEXT:1]",
            "Content on page one.",
            "[TEXT:2]",
            "Content on page six.",
        ])
        citations = {
            "#/texts/1": {"page": 1},
            "#/texts/2": {"page": 6},
        }
        _write_md_and_citations(tmp_converted, stem, md, citations)

        chunks = chunker.chunk_document(stem, DocumentType.UNKNOWN, "test.pdf")
        chunk = chunks[0]

        d = chunk.to_dict()
        reloaded = json.loads(json.dumps(d))

        assert reloaded["citation"]["page_spans"] == chunk.citation["page_spans"]
