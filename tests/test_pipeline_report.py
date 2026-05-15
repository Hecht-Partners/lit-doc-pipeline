"""Tests for generate_pipeline_report() in run_pipeline.py."""

import os
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure test env is set so tqdm is disabled
os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

from lit_pipeline.run_pipeline import generate_pipeline_report, _format_duration


# ── Helpers ───────────────────────────────────────────────────────────

def _make_ok_result(
    file="doc.pdf",
    stem="doc",
    doc_type="exhibit",
    citations_count=50,
    coverage_pct=95.0,
    chunks_count=10,
    extraction_method="docling",
    elapsed_seconds=5.0,
    classification_confidence=0.85,
    classification_needs_input=False,
    had_json=True,
    citation_degraded=False,
    bates_gaps=None,
    bates_duplicates=None,
    line_gaps=None,
):
    return {
        "file": file,
        "stem": stem,
        "status": "OK",
        "doc_type": doc_type,
        "md_file": f"{stem}.md",
        "json_file": f"{stem}.json",
        "citations_count": citations_count,
        "coverage_pct": coverage_pct,
        "type_distribution": {},
        "extraction_method": extraction_method,
        "elapsed_seconds": elapsed_seconds,
        "classification_confidence": classification_confidence,
        "classification_needs_input": classification_needs_input,
        "had_json": had_json,
        "citation_degraded": citation_degraded,
        "chunks_count": chunks_count,
        "bates_gaps": bates_gaps or [],
        "bates_duplicates": bates_duplicates or [],
        "line_gaps": line_gaps or [],
    }


def _make_failed_result(file="bad.pdf", stem="bad", error="Conversion timeout"):
    return {
        "file": file,
        "stem": stem,
        "status": "FAILED",
        "error": error,
        "elapsed_seconds": 300.0,
    }


def _make_skipped_result(file="skip.pdf", stem="skip"):
    return {
        "file": file,
        "stem": stem,
        "status": "SKIPPED",
        "reason": "Already completed",
    }


def _capture_report(**kwargs):
    """Call generate_pipeline_report and capture stdout."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        generate_pipeline_report(**kwargs)
    return buf.getvalue()


def _default_kwargs(results, **overrides):
    """Build kwargs for generate_pipeline_report with sensible defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_dir = Path(tmpdir)
        # Create some dummy files so OUTPUT section works
        for r in results:
            if r["status"] == "OK":
                (converted_dir / f"{r['stem']}.md").write_text("content")
                (converted_dir / f"{r['stem']}_citations.json").write_text("{}")
                (converted_dir / f"{r['stem']}_chunks.json").write_text("[]")

        defaults = {
            "results": results,
            "skipped_condensed": [],
            "converted_dir": converted_dir,
            "pipeline_elapsed": 120.0,
            "cleanup_json": True,
            "parallel": False,
            "max_workers": None,
        }
        defaults.update(overrides)
        return _capture_report(**defaults)


# ── Test: _format_duration ────────────────────────────────────────────

class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(5.3) == "5.3s"

    def test_seconds_zero(self):
        assert _format_duration(0.0) == "0.0s"

    def test_minutes(self):
        assert _format_duration(125) == "2m 05s"

    def test_exact_minute(self):
        assert _format_duration(60) == "1m 00s"

    def test_hours(self):
        assert _format_duration(3661) == "1h 01m"

    def test_large_hours(self):
        assert _format_duration(7200) == "2h 00m"


# ── Test: All-success run ─────────────────────────────────────────────

class TestAllSuccess:
    def test_no_errors_or_warnings_sections(self):
        results = [
            _make_ok_result(file="a.pdf", stem="a", citations_count=100,
                            coverage_pct=95.0, chunks_count=20),
            _make_ok_result(file="b.pdf", stem="b", citations_count=80,
                            coverage_pct=90.0, chunks_count=15),
        ]
        output = _default_kwargs(results)
        assert "PIPELINE REPORT" in output
        assert "SUMMARY" in output
        assert "ERRORS" not in output
        assert "WARNINGS" not in output
        assert "2 OK" in output

    def test_summary_counts(self):
        results = [
            _make_ok_result(file="a.pdf", stem="a", citations_count=100, chunks_count=20),
            _make_ok_result(file="b.pdf", stem="b", citations_count=50, chunks_count=10),
        ]
        output = _default_kwargs(results)
        assert "2 total" in output
        assert "2 OK" in output
        assert "0 failed" in output
        assert "30" in output  # 20 + 10 chunks
        assert "150" in output  # 100 + 50 citations


# ── Test: Failures ────────────────────────────────────────────────────

class TestFailures:
    def test_failures_in_errors_section(self):
        results = [
            _make_ok_result(file="good.pdf", stem="good"),
            _make_failed_result(file="bad1.pdf", stem="bad1",
                                error="Conversion timeout (>1800 seconds)"),
            _make_failed_result(file="bad2.pdf", stem="bad2",
                                error="Expected output not found"),
        ]
        output = _default_kwargs(results)
        assert "ERRORS (2 documents)" in output
        assert "FAIL  bad1.pdf" in output
        assert "Conversion timeout" in output
        assert "FAIL  bad2.pdf" in output
        assert "Expected output not found" in output

    def test_single_failure_singular(self):
        results = [
            _make_failed_result(file="bad.pdf", stem="bad", error="Oops"),
        ]
        output = _default_kwargs(results)
        assert "ERRORS (1 document)" in output


# ── Test: Warning categories ─────────────────────────────────────────

class TestWarnings:
    def test_zero_citations_warning(self):
        results = [
            _make_ok_result(file="empty.pdf", stem="empty", doc_type="exhibit",
                            citations_count=0, coverage_pct=0.0, chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "WARNINGS" in output
        assert "Zero citations" in output
        assert "empty.pdf" in output

    def test_low_coverage_warning(self):
        results = [
            _make_ok_result(file="scan.pdf", stem="scan", doc_type="exhibit",
                            citations_count=12, coverage_pct=23.1, chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "WARNINGS" in output
        assert "Low coverage < 50%" in output
        assert "scan.pdf" in output
        assert "23.1%" in output

    def test_zero_chunks_warning(self):
        results = [
            _make_ok_result(file="blank.pdf", stem="blank", doc_type="exhibit",
                            citations_count=0, chunks_count=0),
        ]
        output = _default_kwargs(results)
        assert "Zero chunks produced" in output
        assert "blank.pdf" in output

    def test_citation_degraded_warning(self):
        results = [
            _make_ok_result(file="partial.pdf", stem="partial",
                            citation_degraded=True, chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "Citation tracking degraded" in output
        assert "partial.pdf" in output

    def test_no_json_warning(self):
        results = [
            _make_ok_result(file="nojson.pdf", stem="nojson",
                            had_json=False, chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "No JSON for citation tracking" in output
        assert "nojson.pdf" in output

    def test_low_confidence_classification_warning(self):
        results = [
            _make_ok_result(file="mystery.pdf", stem="mystery", doc_type="exhibit",
                            classification_confidence=0.03, chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "Low-confidence classification" in output
        assert "mystery.pdf" in output
        assert "0.03" in output

    def test_no_warnings_when_all_good(self):
        results = [
            _make_ok_result(file="good.pdf", stem="good", citations_count=50,
                            coverage_pct=90.0, chunks_count=10,
                            classification_confidence=0.85),
        ]
        output = _default_kwargs(results)
        assert "WARNINGS" not in output


# ── Test: Classification distribution ─────────────────────────────────

class TestClassificationDistribution:
    def test_counts_match(self):
        results = [
            _make_ok_result(file="a.pdf", stem="a", doc_type="exhibit", chunks_count=5),
            _make_ok_result(file="b.pdf", stem="b", doc_type="exhibit", chunks_count=5),
            _make_ok_result(file="c.pdf", stem="c", doc_type="deposition", chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "CLASSIFICATION DISTRIBUTION" in output
        assert "exhibit" in output
        assert "deposition" in output


# ── Test: Extraction methods ──────────────────────────────────────────

class TestExtractionMethods:
    def test_both_methods_shown(self):
        results = [
            _make_ok_result(file="a.pdf", stem="a", extraction_method="docling",
                            chunks_count=5),
            _make_ok_result(file="b.pdf", stem="b", extraction_method="pymupdf",
                            chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "EXTRACTION METHODS" in output
        assert "docling" in output
        assert "pymupdf (fast path)" in output


# ── Test: Timing format ──────────────────────────────────────────────

class TestTiming:
    def test_parallel_mode_shown(self):
        results = [_make_ok_result(chunks_count=5)]
        output = _default_kwargs(results, parallel=True, max_workers=4)
        assert "parallel, 4 workers" in output

    def test_sequential_mode_shown(self):
        results = [_make_ok_result(chunks_count=5)]
        output = _default_kwargs(results, parallel=False)
        assert "sequential" in output


# ── Test: Info section ────────────────────────────────────────────────

class TestInfoSection:
    def test_skipped_shown(self):
        results = [
            _make_skipped_result(file="already.pdf", stem="already"),
            _make_ok_result(file="new.pdf", stem="new", chunks_count=5),
        ]
        output = _default_kwargs(results)
        assert "INFO" in output
        assert "Skipped: 1" in output

    def test_condensed_shown(self):
        results = [_make_ok_result(chunks_count=5)]
        output = _default_kwargs(results,
                                 skipped_condensed=["condensed_transcript.pdf"])
        assert "INFO" in output
        assert "Condensed transcripts: 1" in output
        assert "condensed_transcript.pdf" in output


# ── Test: Empty results ──────────────────────────────────────────────

class TestEmptyResults:
    def test_empty_no_crash(self):
        output = _default_kwargs([])
        assert "PIPELINE REPORT" in output
        assert "0 total" in output
        assert "0 OK" in output

    def test_no_classification_or_extraction_sections_when_empty(self):
        output = _default_kwargs([])
        assert "CLASSIFICATION DISTRIBUTION" not in output
        assert "EXTRACTION METHODS" not in output


# ── Test: Output section ─────────────────────────────────────────────

class TestOutputSection:
    def test_output_directory_shown(self):
        results = [_make_ok_result(file="doc.pdf", stem="doc", chunks_count=5)]
        output = _default_kwargs(results)
        assert "OUTPUT" in output
        assert "Directory:" in output
        assert "Files:" in output
        assert "Size:" in output

    def test_cleanup_json_note(self):
        results = [_make_ok_result(chunks_count=5)]
        output = _default_kwargs(results, cleanup_json=True)
        assert "cleaned up" in output

    def test_no_cleanup_note_when_false(self):
        results = [_make_ok_result(chunks_count=5)]
        output = _default_kwargs(results, cleanup_json=False)
        assert "cleaned up" not in output
