"""Tests for PyMuPDF-based deposition extractor."""

import json
import tempfile
from pathlib import Path

import pytest

from lit_pipeline.pymupdf_extractor import is_text_based_pdf, extract_deposition, _detect_page_marker

TEST_DOCS = Path(__file__).parent / "test_docs"
ALEXANDER_PDF = TEST_DOCS / "Daniel Alexander - 10-24-2025.pdf"
INTEL_PATENT_PDF = TEST_DOCS / "INTEL_PROX_00006214.pdf"


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


class TestIsTextBasedPdf:
    def test_alexander_is_text_based(self):
        assert is_text_based_pdf(str(ALEXANDER_PDF)) is True

    def test_intel_patent_is_scanned(self):
        assert is_text_based_pdf(str(INTEL_PATENT_PDF)) is False

    def test_all_intel_prox_are_scanned(self):
        for name in ["INTEL_PROX_00001770", "INTEL_PROX_00002058",
                      "INTEL_PROX_00002382", "INTEL_PROX_00006214"]:
            pdf = TEST_DOCS / f"{name}.pdf"
            if pdf.exists():
                assert is_text_based_pdf(str(pdf)) is False, f"{name} should be scanned"


class TestExtractDeposition:
    @pytest.fixture(autouse=True)
    def setup(self, output_dir):
        self.result = extract_deposition(str(ALEXANDER_PDF), str(output_dir))
        self.output_dir = output_dir

        # Load citations
        with open(self.result["citations_path"]) as f:
            self.citations = json.load(f)

        # Load markdown
        self.md_content = Path(self.result["md_path"]).read_text()

    def test_produces_md_file(self):
        assert Path(self.result["md_path"]).exists()
        assert self.result["md_path"].endswith(".md")

    def test_produces_citations_file(self):
        assert Path(self.result["citations_path"]).exists()
        assert self.result["citations_path"].endswith("_citations.json")

    def test_has_citations(self):
        assert self.result["citation_count"] > 0
        assert len(self.citations) > 0

    def test_citation_keys_use_page_line_format(self):
        """Citation keys should be like 'line_P14_L5', not Docling self_refs."""
        for key in self.citations:
            assert key.startswith("line_P"), f"Unexpected key format: {key}"
            assert "_L" in key, f"Unexpected key format: {key}"

    def test_exact_line_numbers(self):
        """Each citation should have line_start == line_end (exact, not range)."""
        for key, cit in self.citations.items():
            assert cit["line_start"] == cit["line_end"], (
                f"{key}: line_start={cit['line_start']} != line_end={cit['line_end']}"
            )

    def test_line_numbers_in_valid_range(self):
        """All line numbers should be 1-25."""
        for key, cit in self.citations.items():
            assert 1 <= cit["line_start"] <= 25, (
                f"{key}: line_start={cit['line_start']} out of range"
            )

    def test_transcript_page_present(self):
        """Each citation should have a transcript_page."""
        for key, cit in self.citations.items():
            assert "transcript_page" in cit, f"{key}: missing transcript_page"
            assert cit["transcript_page"] is not None

    def test_page_14_has_expected_content(self):
        """Page 14 should have lines with the deponent's name."""
        page_14_cits = {
            k: v for k, v in self.citations.items()
            if v.get("transcript_page") == 14
        }
        assert len(page_14_cits) > 0, "No citations found for page 14"
        # Line 1 on page 14 should exist
        assert "line_P14_L1" in self.citations

    def test_citation_type_is_transcript_line(self):
        for key, cit in self.citations.items():
            assert cit["type"] == "transcript_line"

    def test_md_contains_page_markers(self):
        """Markdown should contain [PAGE:N] markers."""
        assert "[PAGE:" in self.md_content

    def test_multiple_pages_covered(self):
        """Should cover many transcript pages."""
        pages = {cit["transcript_page"] for cit in self.citations.values()}
        assert len(pages) > 50, f"Only {len(pages)} pages covered"

    def test_lines_per_page_count(self):
        """Most content pages should have close to 25 lines."""
        page_line_counts = {}
        for cit in self.citations.values():
            tp = cit["transcript_page"]
            page_line_counts[tp] = page_line_counts.get(tp, 0) + 1

        # Count pages with 20+ lines (allowing some with fewer due to Q/A breaks)
        full_pages = sum(1 for count in page_line_counts.values() if count >= 20)
        total_pages = len(page_line_counts)
        assert full_pages > total_pages * 0.5, (
            f"Only {full_pages}/{total_pages} pages have 20+ lines"
        )

    def test_no_confidential_header_in_md(self):
        """CONFIDENTIAL header lines should be filtered out of content."""
        header_pattern = "CONFIDENTIAL - OUTSIDE COUNSEL"
        for line in self.md_content.split("\n"):
            if line.startswith("[PAGE:"):
                continue
            assert header_pattern not in line.upper(), (
                f"CONFIDENTIAL header leaked into content: {line[:80]}"
            )


class TestPageMarkerDetection:
    """Test _detect_page_marker correctly requires BOTH right-side AND top position."""

    def _make_span(self, text, x0, y_mid):
        """Create a mock span dict."""
        return {"text": text, "x0": x0, "y_mid": y_mid, "x1": x0 + 50, "y0": y_mid - 5, "y1": y_mid + 5}

    def test_valid_page_marker_top_right(self):
        """Page N at top-right should be detected."""
        spans = [self._make_span("Page 5", x0=450, y_mid=80)]
        assert _detect_page_marker(spans) == 5

    def test_mid_page_right_side_not_marker(self):
        """Page N on the right side but mid-page should NOT be detected."""
        # x0=450 (right side) but y_mid=400 (middle of page)
        spans = [self._make_span("Page 5", x0=450, y_mid=400)]
        assert _detect_page_marker(spans) is None

    def test_top_left_not_marker(self):
        """Page N near top but on the left side should NOT be detected."""
        # x0=50 (left side) but y_mid=80 (top of page)
        spans = [self._make_span("Page 5", x0=50, y_mid=80)]
        assert _detect_page_marker(spans) is None

    def test_non_page_text_not_marker(self):
        """Random text should not be detected as a page marker."""
        spans = [self._make_span("Hello World", x0=450, y_mid=80)]
        assert _detect_page_marker(spans) is None
