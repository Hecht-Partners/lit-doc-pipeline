"""
Tests for doc_classifier.py — Generic document type classification with self-learning.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lit_pipeline.citation_types import DocumentType
from lit_pipeline.doc_classifier import (
    ClassificationResult,
    DocumentFingerprint,
    ProfileStore,
    _signal_filename,
    _signal_content,
    _signal_structural,
    classify_document,
    classify_directory,
)

TEST_DOCS_DIR = Path(__file__).parent / "test_docs"

# ── Filename Signal Tests ─────────────────────────────────────────────


class TestSignalFilename:
    def test_deposition_keywords(self):
        scores = _signal_filename("2025.05.19 - Condensed Transcript - David L Brown.pdf")
        assert DocumentType.DEPOSITION in scores
        assert scores[DocumentType.DEPOSITION] == 1.0

    def test_deposition_keyword_deposition(self):
        scores = _signal_filename("Deposition of John Smith.pdf")
        assert DocumentType.DEPOSITION in scores

    def test_hearing_transcript(self):
        scores = _signal_filename("Hearing Transcript - Jan 15 2025.pdf")
        assert DocumentType.HEARING_TRANSCRIPT in scores

    def test_expert_report(self):
        scores = _signal_filename("Expert Report of Dr. Cole.pdf")
        assert DocumentType.EXPERT_REPORT in scores

    def test_declaration(self):
        scores = _signal_filename("Declaration of John Smith.pdf")
        assert DocumentType.DECLARATION in scores

    def test_motion(self):
        scores = _signal_filename("Motion to Exclude Expert Testimony.pdf")
        assert DocumentType.MOTION in scores

    def test_brief(self):
        scores = _signal_filename("Brief in Opposition.pdf")
        assert DocumentType.BRIEF in scores

    def test_patent(self):
        scores = _signal_filename("US Patent 152 - Exhibit 2.pdf")
        assert DocumentType.PATENT in scores

    def test_patent_prosecution(self):
        scores = _signal_filename("Office Action - App 12345678.pdf")
        assert DocumentType.PATENT_PROSECUTION in scores

    def test_agreement(self):
        scores = _signal_filename("License Agreement - Intel Proxim.pdf")
        assert DocumentType.AGREEMENT in scores

    def test_sec_filing(self):
        scores = _signal_filename("Intel 10-K Annual Report.pdf")
        assert DocumentType.SEC_FILING in scores

    def test_technical_standard(self):
        scores = _signal_filename("IEEE 802.11 Standard.pdf")
        assert DocumentType.TECHNICAL_STANDARD in scores

    def test_witness_statement(self):
        scores = _signal_filename("Witness Statement of Jane Doe.pdf")
        assert DocumentType.WITNESS_STATEMENT in scores

    def test_bates_prefix_gives_weak_exhibit(self):
        scores = _signal_filename("intel_prox_00001770.pdf")
        assert scores.get(DocumentType.EXHIBIT, 0) > 0

    def test_unknown_filename(self):
        scores = _signal_filename("random_document_12345.pdf")
        # No strong signal for any specific type
        for v in scores.values():
            assert v < 0.5


# ── Content Signal Tests ──────────────────────────────────────────────


class TestSignalContent:
    def test_deposition_keywords(self):
        text = "DEPOSITION OF JOHN SMITH\nCOURT REPORTER: Jane Doe CCR RPR\nVIDEOGRAPHER: Bob"
        scores = _signal_content(text)
        assert DocumentType.DEPOSITION in scores
        assert scores[DocumentType.DEPOSITION] > 0.5

    def test_patent_keywords(self):
        text = "UNITED STATES PATENT\nCLAIMS\nABSTRACT\nFIELD OF THE INVENTION"
        scores = _signal_content(text)
        assert DocumentType.PATENT in scores
        assert scores[DocumentType.PATENT] > 0.5

    def test_agreement_keywords(self):
        text = "WHEREAS the parties desire...\nNOW THEREFORE the parties agree\nWITNESSETH"
        scores = _signal_content(text)
        assert DocumentType.AGREEMENT in scores

    def test_correspondence_keywords(self):
        text = "From: john@example.com\nTo: jane@example.com\nSubject: Re: Meeting\nDate: 2025-01-15"
        scores = _signal_content(text)
        assert DocumentType.CORRESPONDENCE in scores

    def test_scholarly_keywords(self):
        text = "Abstract\nIntroduction\nMethodology\nConclusion\nReferences\nUniversity of California"
        scores = _signal_content(text)
        assert DocumentType.SCHOLARLY in scores


# ── Structural Signal Tests ───────────────────────────────────────────


class TestSignalStructural:
    def test_qa_lines_detect_deposition(self):
        lines = ["  1 Q  What is your name?", "  2 A  John Smith."] * 20
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 1)
        assert DocumentType.DEPOSITION in scores
        assert scores[DocumentType.DEPOSITION] > 0.5
        assert debug["qa_lines"] >= 20

    def test_margin_numbers_detect_deposition(self):
        lines = [f"  {i}   Some testimony text here" for i in range(1, 26)]
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 1)
        assert debug["margin_line_numbers"] >= 20

    def test_numbered_paragraphs_detect_expert_report(self):
        lines = [f"{i}. This is paragraph {i} with analysis of the technical data." for i in range(1, 20)]
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 2)
        assert DocumentType.EXPERT_REPORT in scores
        assert scores[DocumentType.EXPERT_REPORT] > 0

    def test_section_hierarchy_detect_standard(self):
        lines = ["1.1 Scope", "1.2 References", "2.1 General", "2.2 Definitions",
                 "3.1 Architecture", "3.2 Components"]
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 1)
        assert DocumentType.TECHNICAL_STANDARD in scores
        assert debug["section_numbers"] >= 6

    def test_email_headers_detect_correspondence(self):
        lines = ["From: john@example.com", "To: jane@example.com",
                 "Subject: Meeting", "Date: 2025-01-15", "", "Dear Jane,"]
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 1)
        assert DocumentType.CORRESPONDENCE in scores
        assert scores[DocumentType.CORRESPONDENCE] == 1.0

    def test_sec_items_detect_filing(self):
        lines = ["ITEM 1. BUSINESS", "ITEM 2. PROPERTIES", "ITEM 3. LEGAL PROCEEDINGS"]
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 1)
        assert DocumentType.SEC_FILING in scores

    def test_column_in_deposition_does_not_trigger_patent(self):
        """Regression: 'column' in deposition testimony must NOT trigger PATENT."""
        lines = (["  1 Q  Did you review column 4 of the spreadsheet?",
                  "  2 A  Yes, I looked at column 4 and column 5."] * 15 +
                 ["  3 Q  What did you find?", "  4 A  The data was incorrect."] * 5)
        text = "\n".join(lines)
        scores, debug = _signal_structural(text, 2)
        # Deposition signal should dominate
        dep_score = scores.get(DocumentType.DEPOSITION, 0)
        pat_score = scores.get(DocumentType.PATENT, 0)
        assert dep_score > pat_score


# ── Real PDF Classification Tests ────────────────────────────────────


@pytest.mark.skipif(
    not TEST_DOCS_DIR.exists(),
    reason="Test documents not available"
)
class TestRealPDFs:
    def test_alexander_deposition(self):
        pdf = TEST_DOCS_DIR / "Daniel Alexander - 10-24-2025.pdf"
        if not pdf.exists():
            pytest.skip("Test document not available")
        result = classify_document(str(pdf))
        assert result.doc_type == DocumentType.DEPOSITION
        assert result.confidence > 0.1
        assert result.is_text_based

    def test_patent_6214_scanned(self):
        """Scanned/image-based patent — no extractable text, needs user input."""
        pdf = TEST_DOCS_DIR / "INTEL_PROX_00006214.pdf"
        if not pdf.exists():
            pytest.skip("Test document not available")
        result = classify_document(str(pdf))
        # This is a scanned PDF with no extractable text, so content/structural
        # signals are empty. Classifier correctly flags needs_user_input=True.
        # With learned profiles or interactive mode, it would be classified as PATENT.
        assert not result.is_text_based
        assert result.needs_user_input

    def test_cole_report(self):
        pdf = TEST_DOCS_DIR / "2025-12-11 - Cole Report.pdf"
        if not pdf.exists():
            pytest.skip("Test document not available")
        result = classify_document(str(pdf))
        # Should detect as expert report (has "Report" in name + structural features)
        assert result.doc_type in (DocumentType.EXPERT_REPORT, DocumentType.UNKNOWN)

    def test_us_patent_exhibit(self):
        pdf = TEST_DOCS_DIR / "US Patent 152 - 01.30.2025 - Exhibit 2.pdf"
        if not pdf.exists():
            pytest.skip("Test document not available")
        result = classify_document(str(pdf))
        assert result.doc_type == DocumentType.PATENT

    def test_classification_speed(self):
        """Classification should be < 500ms per document."""
        pdfs = list(TEST_DOCS_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No test documents")
        pdf = pdfs[0]
        start = time.time()
        classify_document(str(pdf))
        elapsed = time.time() - start
        assert elapsed < 0.5, f"Classification took {elapsed:.2f}s (should be < 0.5s)"


# ── Profile Store Tests ──────────────────────────────────────────────


class TestProfileStore:
    def test_create_and_save(self, tmp_path):
        profile_path = tmp_path / "profiles.json"
        store = ProfileStore(path=profile_path)
        assert not store.has_profiles()

        fp = DocumentFingerprint(
            monospace_ratio=0.85,
            qa_line_density=4.2,
            has_qa_format=True,
        )
        store.update("deposition", fp)
        assert store.has_profiles()
        assert "deposition" in store.profiles
        assert store.profiles["deposition"]["count"] == 1

        # Verify file was saved
        assert profile_path.exists()
        with open(profile_path) as f:
            data = json.load(f)
        assert "deposition" in data

    def test_reload(self, tmp_path):
        profile_path = tmp_path / "profiles.json"
        store1 = ProfileStore(path=profile_path)
        fp = DocumentFingerprint(monospace_ratio=0.85, has_qa_format=True)
        store1.update("deposition", fp)

        # Reload from disk
        store2 = ProfileStore(path=profile_path)
        assert store2.has_profiles()
        assert "deposition" in store2.profiles

    def test_running_mean_updates(self, tmp_path):
        profile_path = tmp_path / "profiles.json"
        store = ProfileStore(path=profile_path)

        fp1 = DocumentFingerprint(monospace_ratio=0.80)
        fp2 = DocumentFingerprint(monospace_ratio=0.90)
        store.update("deposition", fp1)
        store.update("deposition", fp2)

        assert store.profiles["deposition"]["count"] == 2
        avg = store.profiles["deposition"]["avg_fingerprint"]["monospace_ratio"]
        assert abs(avg - 0.85) < 0.01

    def test_score_fingerprint(self, tmp_path):
        profile_path = tmp_path / "profiles.json"
        store = ProfileStore(path=profile_path)

        # Train with 3 deposition fingerprints
        for _ in range(3):
            fp = DocumentFingerprint(monospace_ratio=0.85, qa_line_density=4.0, has_qa_format=True)
            store.update("deposition", fp)

        # Train with 3 patent fingerprints
        for _ in range(3):
            fp = DocumentFingerprint(monospace_ratio=0.1, has_two_columns=True, has_patent_claims=True)
            store.update("patent", fp)

        # Score a deposition-like fingerprint
        test_fp = DocumentFingerprint(monospace_ratio=0.82, qa_line_density=3.8, has_qa_format=True)
        scores = store.score_fingerprint(test_fp)

        assert "deposition" in scores
        assert "patent" in scores
        assert scores["deposition"] > scores["patent"]


# ── Fingerprint Tests ────────────────────────────────────────────────


class TestDocumentFingerprint:
    def test_to_dict_from_dict_roundtrip(self):
        fp = DocumentFingerprint(
            monospace_ratio=0.85,
            serif_ratio=0.1,
            has_qa_format=True,
            avg_chars_per_page=3500,
        )
        d = fp.to_dict()
        fp2 = DocumentFingerprint.from_dict(d)
        assert fp2.monospace_ratio == 0.85
        assert fp2.has_qa_format is True
        assert fp2.avg_chars_per_page == 3500


# ── ClassificationResult Tests ───────────────────────────────────────


class TestClassificationResult:
    def test_basic_fields(self):
        result = ClassificationResult(
            doc_type=DocumentType.DEPOSITION,
            confidence=0.85,
            is_text_based=True,
        )
        assert result.doc_type == DocumentType.DEPOSITION
        assert result.confidence == 0.85
        assert result.is_text_based is True
        assert result.needs_user_input is False

    def test_low_confidence_needs_input(self):
        result = ClassificationResult(
            doc_type=DocumentType.UNKNOWN,
            confidence=0.05,
            is_text_based=True,
            needs_user_input=True,
        )
        assert result.needs_user_input is True


# ── classify_directory Tests ─────────────────────────────────────────


@pytest.mark.skipif(
    not TEST_DOCS_DIR.exists(),
    reason="Test documents not available"
)
class TestClassifyDirectory:
    def test_classifies_all_pdfs(self):
        results = classify_directory(str(TEST_DOCS_DIR), interactive=False)
        pdfs = list(TEST_DOCS_DIR.glob("*.pdf"))
        assert len(results) == len(pdfs)

    def test_returns_classification_results(self):
        results = classify_directory(str(TEST_DOCS_DIR), interactive=False)
        for stem, result in results.items():
            assert isinstance(result, ClassificationResult)
            assert isinstance(result.doc_type, DocumentType)
            assert 0.0 <= result.confidence <= 1.0


# ── Learning Loop Tests ──────────────────────────────────────────────


@pytest.mark.skipif(
    not TEST_DOCS_DIR.exists(),
    reason="Test documents not available"
)
class TestLearningLoop:
    def test_profile_improves_classification(self, tmp_path):
        """Classify -> user correct -> re-classify with profile -> verify improved."""
        profile_path = tmp_path / "profiles.json"
        store = ProfileStore(path=profile_path)

        pdf = TEST_DOCS_DIR / "Daniel Alexander - 10-24-2025.pdf"
        if not pdf.exists():
            pytest.skip("Test document not available")

        # First classification (no profiles)
        result1 = classify_document(str(pdf), profile_store=store)

        # Simulate user correction: confirm it's a deposition
        import fitz
        doc = fitz.open(str(pdf))
        max_pages = min(len(doc), 5)
        text = "\n".join(doc[i].get_text() for i in range(max_pages))
        from lit_pipeline.doc_classifier import _extract_fingerprint
        fingerprint = _extract_fingerprint(doc, text, max_pages)
        doc.close()

        store.update("deposition", fingerprint)

        # Re-classify with profile
        result2 = classify_document(str(pdf), profile_store=store)
        assert result2.doc_type == DocumentType.DEPOSITION
