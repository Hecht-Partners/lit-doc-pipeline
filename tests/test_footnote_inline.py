"""Tests for inline footnote insertion in PostProcessor."""

import json
import tempfile
from pathlib import Path

import pytest

from lit_pipeline.post_processor import PostProcessor
from lit_pipeline.citation_types import DocumentType


@pytest.fixture
def cole_report_files():
    """Return paths to Cole Report test files."""
    base = Path("tests/six_doc_test/converted")
    return {
        "md": base / "2025_12_11_cole_report.md",
        "json": base / "2025_12_11_cole_report.json",
    }


class TestFootnoteInlineInsertion:
    def test_footnotes_detected_from_json(self, cole_report_files):
        """Verify footnotes are detected from Docling JSON."""
        if not cole_report_files["json"].exists():
            pytest.skip("Cole Report JSON not available")

        with open(cole_report_files["json"]) as f:
            data = json.load(f)

        footnotes = [t for t in data.get("texts", []) if t.get("label") == "footnote"]
        assert len(footnotes) > 0, "Should detect footnotes in Cole Report"

    def test_footnotes_moved_inline_in_markdown(self, cole_report_files):
        """Verify footnotes are moved inline with paragraphs."""
        if not cole_report_files["md"].exists():
            pytest.skip("Cole Report markdown not available")

        with open(cole_report_files["md"]) as f:
            content = f.read()

        # Check for inline footnote markers
        assert "[FOOTNOTE" in content, "Should have [FOOTNOTE markers in markdown"

        # Count inline footnotes
        fn_count = content.count("[FOOTNOTE")
        assert fn_count >= 5, f"Should have multiple footnotes inline, found {fn_count}"

    def test_paragraph_25_includes_footnotes(self, cole_report_files):
        """Verify paragraph 25 has its footnotes inline for RAG retrieval."""
        if not cole_report_files["md"].exists():
            pytest.skip("Cole Report markdown not available")

        with open(cole_report_files["md"]) as f:
            content = f.read()

        # Find paragraph 25
        lines = content.split("\n")
        para_25_start = None
        for i, line in enumerate(lines):
            if line.startswith("25. "):
                para_25_start = i
                break

        assert para_25_start is not None, "Should find paragraph 25"

        # Get chunk around paragraph 25 (simulate RAG retrieval)
        chunk_lines = lines[para_25_start : para_25_start + 15]
        chunk = "\n".join(chunk_lines)

        # Verify footnotes are in the chunk
        assert "[FOOTNOTE 1:" in chunk, "Footnote 1 should be inline with paragraph 25"
        assert "[FOOTNOTE 2:" in chunk, "Footnote 2 should be inline with paragraph 25"

        # Verify footnote content
        assert "Peeters Infringement Report" in chunk, "Should include footnote citation content"

    def test_substantive_footnote_preserved(self, cole_report_files):
        """Verify long substantive footnotes are preserved inline."""
        if not cole_report_files["md"].exists():
            pytest.skip("Cole Report markdown not available")

        with open(cole_report_files["md"]) as f:
            content = f.read()

        # Footnote 12 is a long substantive footnote about battery life
        assert "[FOOTNOTE 12:" in content, "Should have footnote 12"

        # Check it contains substantive content (not just a citation)
        fn12_start = content.index("[FOOTNOTE 12:")
        fn12_end = content.index("]", fn12_start)
        fn12_content = content[fn12_start:fn12_end]

        assert len(fn12_content) > 200, "Substantive footnote should be long"
        assert "Battery life" in fn12_content, "Should preserve substantive content"
        assert "purchasing decisions" in fn12_content, "Should preserve full argument"

    def test_footnotes_not_duplicated(self, cole_report_files):
        """Verify footnotes appear only once (inline, not in original position)."""
        if not cole_report_files["md"].exists():
            pytest.skip("Cole Report markdown not available")

        with open(cole_report_files["md"]) as f:
            lines = f.readlines()

        # Check that standalone footnote lines (original position) are removed
        standalone_footnotes = [
            line for line in lines
            if line.strip().startswith("1 See The Peeters")
        ]

        # Should not find the original standalone line
        assert len(standalone_footnotes) == 0, (
            "Original standalone footnote lines should be removed"
        )

    def test_postprocessor_integration(self, tmp_path):
        """Test PostProcessor with expert report document type."""
        # Create minimal test files
        test_md = tmp_path / "test.md"
        test_json = tmp_path / "test.json"

        # Minimal expert report markdown
        test_md.write_text(
            "## Expert Report\n\n"
            "¶1. This is a test paragraph.\n\n"
            "1 This is footnote one.\n\n"
            "¶2. Next paragraph.\n"
        )

        # Minimal JSON with footnote
        test_json.write_text(json.dumps({
            "texts": [
                {"label": "section_header", "text": "Expert Report"},
                {"label": "text", "text": "¶1. This is a test paragraph."},
                {"label": "footnote", "text": "1 This is footnote one."},
                {"label": "text", "text": "¶2. Next paragraph."}
            ]
        }))

        # Process with PostProcessor
        processor = PostProcessor()
        result = processor.process(str(test_md), DocumentType.EXPERT_REPORT)

        # Verify footnote was moved inline
        processed = test_md.read_text()
        assert "[FOOTNOTE 1:" in processed, "Should insert footnote inline"
        assert "This is footnote one" in processed, "Should preserve footnote content"
