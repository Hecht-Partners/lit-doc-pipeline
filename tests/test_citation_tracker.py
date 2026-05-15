"""
Tests for citation_tracker.py.

Unit tests use mock data matching actual Docling output structure.
Integration tests run against real test data in tests/processed_20260209_232538/converted/.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lit_pipeline.citation_tracker import CitationTracker, PageLayout, reconstruct_citations
from lit_pipeline.citation_types import DocumentType


# ── Helpers ──────────────────────────────────────────────────────────────

def make_text_elem(
    self_ref: str,
    text: str,
    page_no: int,
    bbox_l: float,
    bbox_t: float,
    bbox_r: float,
    bbox_b: float,
    label: str = "text",
):
    """Create a Docling text element matching real JSON structure."""
    return {
        "self_ref": self_ref,
        "label": label,
        "text": text,
        "prov": [
            {
                "page_no": page_no,
                "bbox": {"l": bbox_l, "t": bbox_t, "r": bbox_r, "b": bbox_b},
            }
        ],
    }


def make_page_marker(page_num: int, pdf_page: int, self_ref: str):
    """Create a Page N marker element (upper-right position)."""
    return make_text_elem(
        self_ref=self_ref,
        text=f"Page {page_num}",
        page_no=pdf_page,
        bbox_l=475.2,
        bbox_t=710.1,
        bbox_r=522.0,
        bbox_b=695.4,
    )


def make_header(text: str, page_no: int, self_ref: str):
    """Create a CONFIDENTIAL header element."""
    return make_text_elem(
        self_ref=self_ref,
        text=text,
        page_no=page_no,
        bbox_l=181.1,
        bbox_t=735.6,
        bbox_r=433.1,
        bbox_b=724.8,
        label="section_header",
    )


def write_mock_json(tmpdir: Path, stem: str, texts: list, pages: dict = None):
    """Write a mock Docling JSON file."""
    doc = {
        "schema_name": "DoclingDocument",
        "version": "1.9.0",
        "texts": texts,
        "pages": pages or {"1": {"size": {"width": 612.0, "height": 792.0}, "page_no": 1}},
    }
    path = tmpdir / f"{stem}.json"
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


def write_mock_bates(tmpdir: Path, stem: str, bates: dict):
    """Write a mock Bates sidecar file."""
    path = tmpdir / f"{stem}_bates.json"
    with open(path, "w") as f:
        json.dump(bates, f)
    return path


# ── PageLayout Tests ─────────────────────────────────────────────────────

class TestPageLayout:
    def test_line_height_calculation(self):
        layout = PageLayout(content_top=655.0, content_bottom=87.5, lines_per_page=25)
        assert abs(layout.line_height - 22.7) < 0.1

    def test_bbox_to_lines_top_of_page(self):
        layout = PageLayout(content_top=655.0, content_bottom=87.5, lines_per_page=25)
        line_start, line_end = layout.bbox_to_lines(655.0, 640.0)
        assert line_start == 1

    def test_bbox_to_lines_bottom_of_page(self):
        layout = PageLayout(content_top=655.0, content_bottom=87.5, lines_per_page=25)
        line_start, line_end = layout.bbox_to_lines(110.0, 87.5)
        assert line_end == 25

    def test_bbox_to_lines_clamped(self):
        layout = PageLayout(content_top=655.0, content_bottom=87.5, lines_per_page=25)
        # bbox above content area
        line_start, line_end = layout.bbox_to_lines(700.0, 660.0)
        assert line_start == 1
        # bbox below content area
        line_start, line_end = layout.bbox_to_lines(50.0, 10.0)
        assert line_end == 25


# ── Deposition Unit Tests ────────────────────────────────────────────────

class TestDepositionLineInference:
    def test_basic_10_elements(self, tmp_path):
        """10 content elements on 1 page, verify lines in [1, 25]."""
        texts = [make_page_marker(8, 1, "#/texts/0")]
        # Simulate 10 content lines evenly spaced across content area
        for j in range(10):
            t = 655.0 - j * 55.0
            b = t - 24.0
            texts.append(
                make_text_elem(
                    self_ref=f"#/texts/{j+1}",
                    text=f"Q      Some question {j}?" if j % 2 == 0 else f"A      Some answer {j}.",
                    page_no=1,
                    bbox_l=108.0,
                    bbox_t=t,
                    bbox_r=520.0,
                    bbox_b=b,
                )
            )

        write_mock_json(tmp_path, "test_dep", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_dep")

        # Should have 10 citations (page marker is skipped)
        assert len(citations) == 10
        for ref, cit in citations.items():
            assert 1 <= cit["line_start"] <= 25
            assert 1 <= cit["line_end"] <= 25
            assert cit["line_start"] <= cit["line_end"]
            assert cit["type"] == "transcript_line"

    def test_transcript_page_tracking(self, tmp_path):
        """2 pages with markers, verify correct page assignment."""
        texts = [
            make_page_marker(7, 1, "#/texts/0"),
            make_text_elem("#/texts/1", "Q   Question on page 7", 1, 108.0, 600.0, 500.0, 580.0),
            make_text_elem("#/texts/2", "A   Answer on page 7", 1, 108.0, 500.0, 500.0, 480.0),
            make_page_marker(8, 2, "#/texts/3"),
            make_text_elem("#/texts/4", "Q   Question on page 8", 2, 108.0, 600.0, 500.0, 580.0),
        ]

        pages = {
            "1": {"size": {"width": 612.0, "height": 792.0}, "page_no": 1},
            "2": {"size": {"width": 612.0, "height": 792.0}, "page_no": 2},
        }
        write_mock_json(tmp_path, "test_dep2", texts, pages)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_dep2")

        assert citations["#/texts/1"]["transcript_page"] == 7
        assert citations["#/texts/2"]["transcript_page"] == 7
        assert citations["#/texts/4"]["transcript_page"] == 8

    def test_code_block_parsing(self, tmp_path):
        """Embedded line numbers in code blocks are extracted."""
        texts = [
            make_page_marker(47, 1, "#/texts/0"),
            make_text_elem(
                "#/texts/1",
                "1 DANIEL ALEXANDER 2 Do you know what the H 3 processor line designation means?",
                1,
                108.0,
                655.0,
                520.0,
                500.0,
                label="code",
            ),
        ]
        write_mock_json(tmp_path, "test_code", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_code")

        assert "#/texts/1" in citations
        cit = citations["#/texts/1"]
        assert cit["line_start"] == 1
        assert cit["line_end"] == 3
        assert cit["type"] == "transcript_line"

    def test_bates_association(self, tmp_path):
        """Bates sidecar lookup works."""
        texts = [
            make_page_marker(5, 1, "#/texts/0"),
            make_text_elem("#/texts/1", "Some text", 1, 108.0, 600.0, 500.0, 580.0),
            make_text_elem("#/texts/2", "More text", 2, 108.0, 600.0, 500.0, 580.0),
        ]
        pages = {
            "1": {"size": {"width": 612.0, "height": 792.0}, "page_no": 1},
            "2": {"size": {"width": 612.0, "height": 792.0}, "page_no": 2},
        }
        write_mock_json(tmp_path, "test_bates", texts, pages)
        write_mock_bates(tmp_path, "test_bates", {"1": "BATES_00001", "2": "BATES_00002"})

        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_bates")

        assert citations["#/texts/1"]["bates"] == "BATES_00001"
        assert citations["#/texts/2"]["bates"] == "BATES_00002"

    def test_header_exclusion(self, tmp_path):
        """Headers are not included in output."""
        texts = [
            make_header("CONFIDENTIAL - OUTSIDE COUNSEL'S EYES ONLY", 1, "#/texts/0"),
            make_page_marker(5, 1, "#/texts/1"),
            make_text_elem("#/texts/2", "A      Yes.", 1, 108.0, 400.0, 300.0, 385.0),
        ]
        write_mock_json(tmp_path, "test_hdr", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_hdr")

        assert "#/texts/0" not in citations
        assert "#/texts/2" in citations

    def test_page_marker_vs_conversational(self, tmp_path):
        """'Page 18' in Q/A text (low bbox.l) is NOT treated as marker."""
        texts = [
            make_page_marker(10, 1, "#/texts/0"),
            # Conversational "Page 18" at content position (l=108)
            make_text_elem(
                "#/texts/1",
                "Page 18",
                1,
                108.0,
                400.0,
                200.0,
                385.0,
            ),
            make_text_elem("#/texts/2", "A   Some answer", 1, 108.0, 300.0, 500.0, 280.0),
        ]
        write_mock_json(tmp_path, "test_conv", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_conv")

        # The conversational "Page 18" should be in citations as a regular element
        assert "#/texts/1" in citations
        # And the transcript page should still be 10 (from the real marker)
        assert citations["#/texts/1"]["transcript_page"] == 10
        assert citations["#/texts/2"]["transcript_page"] == 10


# ── Patent Unit Tests ────────────────────────────────────────────────────

class TestPatentColumnDetection:
    def _make_spec_page_texts(self, page_no=1):
        """Create enough elements on a page to qualify as a spec page."""
        texts = []
        for j in range(6):
            t = 700.0 - j * 80.0
            texts.append(
                make_text_elem(
                    f"#/texts/filler_{page_no}_{j}",
                    f"Additional specification text for testing column detection on page {page_no} element {j}.",
                    page_no,
                    60.0,
                    t,
                    250.0,
                    t - 60.0,
                )
            )
        return texts

    def _make_patent_tracker(self, tmp_path, texts, pages=None):
        if pages is None:
            pages = {"1": {"size": {"width": 612.0, "height": 792.0}, "page_no": 1}}
        write_mock_json(tmp_path, "test_patent", texts, pages)
        return CitationTracker(str(tmp_path), DocumentType.PATENT)

    def test_left_column(self, tmp_path):
        """bbox.l=60 -> left column."""
        # Need enough elements to qualify page as spec page
        texts = self._make_spec_page_texts(page_no=1) + [
            make_text_elem(
                "#/texts/target",
                "Left column text about the invention and its various aspects and implementations",
                1,
                60.0,
                200.0,
                250.0,
                180.0,
            ),
        ]
        tracker = self._make_patent_tracker(tmp_path, texts)
        citations = tracker.reconstruct_citations("test_patent")

        assert "#/texts/target" in citations
        cit = citations["#/texts/target"]
        assert cit["column"] is not None
        # Column should be odd (left column)
        assert cit["column"] % 2 == 1

    def test_right_column(self, tmp_path):
        """bbox.l=265 -> right column."""
        texts = self._make_spec_page_texts(page_no=1) + [
            make_text_elem(
                "#/texts/target",
                "Right column text about the invention and its various methods and apparatus",
                1,
                265.0,
                200.0,
                470.0,
                180.0,
            ),
        ]
        tracker = self._make_patent_tracker(tmp_path, texts)
        citations = tracker.reconstruct_citations("test_patent")

        assert "#/texts/target" in citations
        cit = citations["#/texts/target"]
        assert cit["column"] is not None
        # Column should be even (right column)
        assert cit["column"] % 2 == 0

    def test_fullwidth_cross_column(self, tmp_path):
        """Full-width element (w > 0.65 * page_width) -> cross_column_merge."""
        texts = self._make_spec_page_texts(page_no=1) + [
            make_text_elem(
                "#/texts/target",
                "This is a very wide element spanning across both columns of the patent document with lots of text",
                1,
                50.0,
                200.0,
                500.0,
                180.0,
            ),
        ]
        tracker = self._make_patent_tracker(tmp_path, texts)
        citations = tracker.reconstruct_citations("test_patent")

        assert "#/texts/target" in citations
        assert citations["#/texts/target"]["type"] == "cross_column_merge"


# ── Expert Report Tests ──────────────────────────────────────────────────

class TestExpertReport:
    def test_paragraph_persistence(self, tmp_path):
        """Paragraph number persists across elements until next marker."""
        texts = [
            make_text_elem("#/texts/0", "¶ 42 The expert states that...", 1, 108.0, 600.0, 500.0, 580.0),
            make_text_elem("#/texts/1", "Continuing discussion of this topic.", 1, 108.0, 560.0, 500.0, 540.0),
            make_text_elem("#/texts/2", "¶ 43 A new paragraph begins here.", 1, 108.0, 520.0, 500.0, 500.0),
        ]
        write_mock_json(tmp_path, "test_expert", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.EXPERT_REPORT)
        citations = tracker.reconstruct_citations("test_expert")

        assert citations["#/texts/0"]["paragraph_number"] == 42
        assert citations["#/texts/1"]["paragraph_number"] == 42  # persists
        assert citations["#/texts/2"]["paragraph_number"] == 43

    def test_paragraph_word_format(self, tmp_path):
        """'Paragraph N' text format is detected."""
        texts = [
            make_text_elem("#/texts/0", "Paragraph 15 discusses the methodology.", 1, 108.0, 600.0, 500.0, 580.0),
            make_text_elem("#/texts/1", "Further analysis follows.", 1, 108.0, 560.0, 500.0, 540.0),
        ]
        write_mock_json(tmp_path, "test_expert2", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.EXPERT_REPORT)
        citations = tracker.reconstruct_citations("test_expert2")

        assert citations["#/texts/0"]["paragraph_number"] == 15
        assert citations["#/texts/1"]["paragraph_number"] == 15


# ── Self-Ref and Validation Tests ────────────────────────────────────────

class TestSelfRefAndValidation:
    def test_self_ref_used_as_key(self, tmp_path):
        """Output keys are self_ref strings, not array indices."""
        # Simulate real Docling where texts[0] has self_ref="#/texts/25"
        texts = [
            {
                "self_ref": "#/texts/25",
                "label": "text",
                "text": "Some content",
                "prov": [{"page_no": 1, "bbox": {"l": 108.0, "t": 600.0, "r": 500.0, "b": 580.0}}],
            },
            {
                "self_ref": "#/texts/26",
                "label": "text",
                "text": "More content",
                "prov": [{"page_no": 1, "bbox": {"l": 108.0, "t": 560.0, "r": 500.0, "b": 540.0}}],
            },
        ]
        write_mock_json(tmp_path, "test_ref", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_ref")

        assert "#/texts/25" in citations
        assert "#/texts/26" in citations
        assert "#/texts/0" not in citations
        assert "#/texts/1" not in citations

    def test_validate_metrics(self, tmp_path):
        """Validate returns correct coverage percentages."""
        texts = [
            make_page_marker(1, 1, "#/texts/0"),
            make_text_elem("#/texts/1", "Content A", 1, 108.0, 600.0, 500.0, 580.0),
            make_text_elem("#/texts/2", "Content B", 1, 108.0, 500.0, 500.0, 480.0),
            make_text_elem("#/texts/3", "Content C", 1, 108.0, 400.0, 500.0, 380.0),
        ]
        write_mock_json(tmp_path, "test_val", texts)
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_val")

        metrics = tracker.validate(citations)
        assert metrics.total_items == 3
        assert metrics.coverage_pct == 100.0
        assert metrics.has_line_numbers is True
        assert metrics.has_bates is False
        assert "transcript_line" in metrics.type_distribution

    def test_missing_json_graceful(self, tmp_path):
        """Missing JSON returns empty dict."""
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("nonexistent")
        assert citations == {}

    def test_empty_texts_array(self, tmp_path):
        """Empty texts array returns empty dict."""
        write_mock_json(tmp_path, "test_empty", [])
        tracker = CitationTracker(str(tmp_path), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("test_empty")
        assert citations == {}


# ── Convenience Function Test ────────────────────────────────────────────

class TestConvenienceFunction:
    def test_reconstruct_citations_function(self, tmp_path):
        """The module-level function works."""
        texts = [
            make_page_marker(1, 1, "#/texts/0"),
            make_text_elem("#/texts/1", "Content", 1, 108.0, 600.0, 500.0, 580.0),
        ]
        write_mock_json(tmp_path, "test_func", texts)
        citations = reconstruct_citations(
            str(tmp_path), "test_func", DocumentType.DEPOSITION
        )
        assert len(citations) == 1


# ── Integration Tests (Real Data) ────────────────────────────────────────

REAL_DATA_DIR = Path(__file__).parent / "processed_20260209_232538" / "converted"


@pytest.mark.skipif(
    not REAL_DATA_DIR.exists(),
    reason="Real test data not available",
)
class TestAlexanderDepositionIntegration:
    """Integration tests against the real Alexander deposition."""

    def test_produces_citations(self):
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        # Should produce a substantial number of citations
        assert len(citations) >= 100, f"Expected >= 100 citations, got {len(citations)}"

    def test_all_lines_valid(self):
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        for ref, cit in citations.items():
            if cit.get("line_start") is not None:
                assert 1 <= cit["line_start"] <= 25, f"{ref}: line_start={cit['line_start']}"
                assert 1 <= cit["line_end"] <= 25, f"{ref}: line_end={cit['line_end']}"
                assert cit["line_start"] <= cit["line_end"], (
                    f"{ref}: line_start={cit['line_start']} > line_end={cit['line_end']}"
                )

    def test_transcript_pages_detected(self):
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        transcript_pages = set()
        for cit in citations.values():
            tp = cit.get("transcript_page")
            if tp is not None:
                transcript_pages.add(tp)

        assert len(transcript_pages) >= 100, (
            f"Expected >= 100 unique transcript pages, got {len(transcript_pages)}"
        )

    def test_self_refs_are_docling_refs(self):
        """All keys should be Docling self_ref format, not array indices."""
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        for ref in citations:
            assert ref.startswith("#/texts/"), f"Unexpected key format: {ref}"

    def test_page8_line_assignments(self):
        """Spot-check: 'Have you been deposed before?' on page 8 around line 8-10."""
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        # Load original JSON to find the element
        with open(REAL_DATA_DIR / "daniel_alexander_10_24_2025.json") as f:
            doc = json.load(f)

        # Find "Have you been deposed before?" element
        target_ref = None
        for t in doc["texts"]:
            if "been deposed before" in t.get("text", ""):
                target_ref = t["self_ref"]
                break

        assert target_ref is not None, "Could not find 'been deposed before' element"
        assert target_ref in citations, f"{target_ref} not in citations"

        cit = citations[target_ref]
        # Should be around lines 6-10 based on bbox position
        assert cit["line_start"] <= 12, f"line_start={cit['line_start']} too high"
        assert cit["transcript_page"] == 8

    def test_validation_metrics(self):
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        metrics = tracker.validate(citations)
        assert metrics.total_items >= 100
        assert metrics.coverage_pct > 90.0
        assert metrics.has_line_numbers is True
        assert metrics.type_distribution.get("transcript_line", 0) > 50

    def test_output_file_created(self):
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.DEPOSITION)
        tracker.reconstruct_citations("daniel_alexander_10_24_2025")

        out_path = REAL_DATA_DIR / "daniel_alexander_10_24_2025_citations.json"
        assert out_path.exists()

        with open(out_path) as f:
            data = json.load(f)
        assert len(data) >= 100


@pytest.mark.skipif(
    not REAL_DATA_DIR.exists(),
    reason="Real test data not available",
)
class TestPatentIntegration:
    """Integration tests against real patent data."""

    def test_patent_6214_produces_citations(self):
        """US Patent (intel_prox_00006214) should have column detection."""
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.PATENT)
        citations = tracker.reconstruct_citations("intel_prox_00006214")

        assert len(citations) > 50, f"Expected > 50 citations, got {len(citations)}"

    def test_patent_6214_has_bates(self):
        """Bates stamps should be populated from sidecar."""
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.PATENT)
        citations = tracker.reconstruct_citations("intel_prox_00006214")

        bates_count = sum(1 for c in citations.values() if c.get("bates"))
        assert bates_count > 0, "No Bates stamps found"

    def test_patent_1770_produces_citations(self):
        """IEEE standard (intel_prox_00001770) should produce citations."""
        tracker = CitationTracker(str(REAL_DATA_DIR), DocumentType.PATENT)
        citations = tracker.reconstruct_citations("intel_prox_00001770")

        assert len(citations) > 100, f"Expected > 100 citations, got {len(citations)}"

        # Should have Bates stamps
        bates_count = sum(1 for c in citations.values() if c.get("bates"))
        assert bates_count > 0, "No Bates stamps found"


# ── New Tests for Citation Improvements ──────────────────────────────


class TestNumberedParagraphDetection:
    """Tests for numbered paragraph format (1. , 2. , 3. )."""

    def test_numbered_paragraph_pattern(self):
        """Detect numbered paragraphs like '1. I, Eric Cole...'"""
        texts = [
            make_text_elem("#/texts/0", "1. I, Eric Cole, have been asked...", 1, 50, 700, 550, 650),
            make_text_elem("#/texts/1", "This is more content for paragraph 1.", 1, 50, 640, 550, 620),
            make_text_elem("#/texts/2", "2. The details of my education...", 1, 50, 610, 550, 590),
        ]

        tracker = CitationTracker(
            converted_dir=tempfile.mkdtemp(),
            doc_type=DocumentType.EXPERT_REPORT
        )
        citations = tracker._handle_expert_report(texts, {0: "#/texts/0", 1: "#/texts/1", 2: "#/texts/2"}, {})

        # First element should have paragraph 1
        assert citations["#/texts/0"]["paragraph_number"] == 1
        assert citations["#/texts/0"]["type"] == "paragraph"

        # Second element should inherit paragraph 1
        assert citations["#/texts/1"]["paragraph_number"] == 1

        # Third element should have paragraph 2
        assert citations["#/texts/2"]["paragraph_number"] == 2

    def test_symbol_paragraph_still_works(self):
        """Ensure ¶ symbol paragraphs still work."""
        texts = [
            make_text_elem("#/texts/0", "¶ 42. The technology enables...", 1, 50, 700, 550, 650),
        ]

        tracker = CitationTracker(
            converted_dir=tempfile.mkdtemp(),
            doc_type=DocumentType.EXPERT_REPORT
        )
        citations = tracker._handle_expert_report(texts, {0: "#/texts/0"}, {})

        assert citations["#/texts/0"]["paragraph_number"] == 42
        assert citations["#/texts/0"]["type"] == "paragraph"

    def test_paragraph_word_format(self):
        """Ensure 'Paragraph N' format still works."""
        texts = [
            make_text_elem("#/texts/0", "Paragraph 15. Background information...", 1, 50, 700, 550, 650),
        ]

        tracker = CitationTracker(
            converted_dir=tempfile.mkdtemp(),
            doc_type=DocumentType.EXPERT_REPORT
        )
        citations = tracker._handle_expert_report(texts, {0: "#/texts/0"}, {})

        assert citations["#/texts/0"]["paragraph_number"] == 15


class TestBatesSequentialValidation:
    """Tests for Bates stamp sequential validation."""

    def test_sequential_bates_no_gaps(self):
        """Sequential Bates stamps should report no gaps."""
        citations = {
            "#/texts/0": {"page": 1, "bates": "INTEL_PROX_00001234", "type": "page_only"},
            "#/texts/1": {"page": 2, "bates": "INTEL_PROX_00001235", "type": "page_only"},
            "#/texts/2": {"page": 3, "bates": "INTEL_PROX_00001236", "type": "page_only"},
        }

        tracker = CitationTracker(
            converted_dir=tempfile.mkdtemp(),
            doc_type=DocumentType.PATENT
        )
        metrics = tracker.validate(citations)

        assert len(metrics.bates_gaps) == 0
        assert len(metrics.bates_duplicates) == 0

    def test_bates_gap_detected(self):
        """Large gap in Bates stamps should be flagged."""
        citations = {
            "#/texts/0": {"page": 1, "bates": "INTEL_PROX_00001234", "type": "page_only"},
            "#/texts/1": {"page": 2, "bates": "INTEL_PROX_00001235", "type": "page_only"},
            "#/texts/2": {"page": 3, "bates": "INTEL_PROX_00001250", "type": "page_only"},  # Gap of 15!
        }

        tracker = CitationTracker(
            converted_dir=tempfile.mkdtemp(),
            doc_type=DocumentType.PATENT
        )
        metrics = tracker.validate(citations)

        assert len(metrics.bates_gaps) > 0
        assert "gap" in metrics.bates_gaps[0].lower()

    def test_bates_duplicates_detected(self):
        """Multiple different Bates on same page should be flagged."""
        citations = {
            "#/texts/0": {"page": 1, "bates": "INTEL_PROX_00001234", "type": "page_only"},
            "#/texts/1": {"page": 1, "bates": "INTEL_PROX_00001235", "type": "page_only"},  # Different Bates on same page
        }

        tracker = CitationTracker(
            converted_dir=tempfile.mkdtemp(),
            doc_type=DocumentType.PATENT
        )
        metrics = tracker.validate(citations)

        assert len(metrics.bates_duplicates) > 0
