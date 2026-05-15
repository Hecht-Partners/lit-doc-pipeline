"""Tests for text_extractor — plain-text court reporter transcript support."""

import json

import pytest

from lit_pipeline.text_extractor import is_text_transcript, extract_text_deposition, _normalize_stem


# ── Helpers ──────────────────────────────────────────────────────────

def _write(tmp_path, name, content):
    """Write *content* to tmp_path/name and return the path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


SAMPLE_TRANSCRIPT = """\
                              1
 1   Q.  Did you review the document?
 2   A.  Yes, I did.
 3   Q.  And what did you find?
 4   A.  It was a technical specification.
 5       It described the wireless protocol.
                              2
 1   Q.  Can you describe the protocol?
 2   A.  It uses TWT for power saving.
 3   Q.  What is TWT?
 4   A.  Target Wake Time.
 5       It allows devices to negotiate wake schedules.
"""


# ── is_text_transcript ───────────────────────────────────────────────

class TestIsTextTranscript:
    def test_valid_transcript(self, tmp_path):
        p = _write(tmp_path, "depo.txt", SAMPLE_TRANSCRIPT)
        assert is_text_transcript(str(p)) is True

    def test_plain_prose(self, tmp_path):
        p = _write(tmp_path, "essay.txt",
                   "This is an essay about wireless networking.\n"
                   "It does not have line numbers or page numbers.\n"
                   "Just plain prose text that should not be detected.\n")
        assert is_text_transcript(str(p)) is False

    def test_only_line_numbers_no_pages(self, tmp_path):
        content = " 1   Some text\n 2   More text\n 3   Even more\n"
        p = _write(tmp_path, "partial.txt", content)
        assert is_text_transcript(str(p)) is False

    def test_only_page_numbers_no_lines(self, tmp_path):
        content = "                              1\nSome text\nMore text\n"
        p = _write(tmp_path, "pages_only.txt", content)
        assert is_text_transcript(str(p)) is False

    def test_nonexistent_file(self):
        assert is_text_transcript("/tmp/does_not_exist_abc123.txt") is False

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert is_text_transcript(str(p)) is False


# ── extract_text_deposition ──────────────────────────────────────────

class TestExtractTextDeposition:
    def test_basic_extraction(self, tmp_path):
        src = _write(tmp_path, "depo.txt", SAMPLE_TRANSCRIPT)
        out_dir = tmp_path / "output"

        result = extract_text_deposition(str(src), str(out_dir))

        assert result["page_count"] == 2
        assert result["line_count"] == 10  # 5 lines per page
        assert result["citation_count"] == 10

        # Check files exist
        assert (out_dir / "depo.md").exists()
        assert (out_dir / "depo_citations.json").exists()

    def test_md_has_page_markers(self, tmp_path):
        src = _write(tmp_path, "depo.txt", SAMPLE_TRANSCRIPT)
        out_dir = tmp_path / "output"

        extract_text_deposition(str(src), str(out_dir))
        md_text = (out_dir / "depo.md").read_text()

        assert "[PAGE:1]" in md_text
        assert "[PAGE:2]" in md_text

    def test_qa_dots_stripped(self, tmp_path):
        src = _write(tmp_path, "depo.txt", SAMPLE_TRANSCRIPT)
        out_dir = tmp_path / "output"

        extract_text_deposition(str(src), str(out_dir))
        md_text = (out_dir / "depo.md").read_text()

        # "Q." and "A." should become "Q" and "A" (no dot)
        assert "Q.  " not in md_text
        assert "A.  " not in md_text
        # But Q and A markers should still be present
        assert "Q  Did you review" in md_text
        assert "A  Yes, I did" in md_text

    def test_citation_key_format(self, tmp_path):
        src = _write(tmp_path, "depo.txt", SAMPLE_TRANSCRIPT)
        out_dir = tmp_path / "output"

        extract_text_deposition(str(src), str(out_dir))

        with open(out_dir / "depo_citations.json") as f:
            citations = json.load(f)

        # Keys should match PyMuPDF format: line_P{page}_L{line}
        assert "line_P1_L1" in citations
        assert "line_P1_L5" in citations
        assert "line_P2_L1" in citations
        assert "line_P2_L5" in citations

        # Check citation data structure
        c = citations["line_P1_L1"]
        assert c["page"] == 1
        assert c["line_start"] == 1
        assert c["line_end"] == 1
        assert c["type"] == "transcript_line"
        assert c["transcript_page"] == 1

    def test_stem_normalization_in_output(self, tmp_path):
        src = _write(tmp_path, "John Doe - 01-15-2026.txt", SAMPLE_TRANSCRIPT)
        out_dir = tmp_path / "output"

        result = extract_text_deposition(str(src), str(out_dir))

        assert (out_dir / "john_doe_01_15_2026.md").exists()
        assert (out_dir / "john_doe_01_15_2026_citations.json").exists()

    def test_empty_file(self, tmp_path):
        src = tmp_path / "empty.txt"
        src.write_text("", encoding="utf-8")
        out_dir = tmp_path / "output"

        result = extract_text_deposition(str(src), str(out_dir))

        assert result["page_count"] == 0
        assert result["line_count"] == 0
        assert result["citation_count"] == 0

    def test_blank_lines_skipped(self, tmp_path):
        content = """\
                              10
 1   Q.  First question?
 2
 3   A.  Answer here.
"""
        src = _write(tmp_path, "depo.txt", content)
        out_dir = tmp_path / "output"

        result = extract_text_deposition(str(src), str(out_dir))

        # Line 2 is blank so should be skipped
        assert result["line_count"] == 2
        assert result["citation_count"] == 2

    def test_continuation_lines(self, tmp_path):
        """Non-Q/A continuation lines should be preserved."""
        content = """\
                              5
 1   Q.  Did you review the document that was
 2       marked as Exhibit A during your earlier
 3       deposition?
 4   A.  Yes.
"""
        src = _write(tmp_path, "depo.txt", content)
        out_dir = tmp_path / "output"

        result = extract_text_deposition(str(src), str(out_dir))
        assert result["line_count"] == 4

        md_text = (out_dir / "depo.md").read_text()
        # Continuation lines should not have Q/A dots stripped
        assert "marked as Exhibit A" in md_text


# ── _normalize_stem ──────────────────────────────────────────────────

class TestNormalizeStem:
    def test_spaces_to_underscores(self):
        assert _normalize_stem("John Doe Deposition") == "john_doe_deposition"

    def test_hyphens_to_underscores(self):
        assert _normalize_stem("kindler-10-24-2025") == "kindler_10_24_2025"

    def test_collapse_multiple_underscores(self):
        assert _normalize_stem("foo___bar") == "foo_bar"

    def test_strip_leading_trailing(self):
        assert _normalize_stem("-foo-") == "foo"

    def test_mixed(self):
        assert _normalize_stem("Daniel Alexander - 10-24-2025") == "daniel_alexander_10_24_2025"
