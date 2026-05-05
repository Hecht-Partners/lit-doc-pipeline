import json
from pathlib import Path
from types import SimpleNamespace

from citation_types import DocumentType
from parallel_processor import process_single_document


class _DummyPostProcessor:
    def process(self, md_path: str, doc_type: DocumentType):
        return SimpleNamespace(citation_coverage=0)


class _DummyCitationTracker:
    def __init__(self, converted_dir: str, doc_type: DocumentType):
        self.converted_dir = converted_dir
        self.doc_type = doc_type

    def reconstruct_citations(self, normalized: str):
        return {}

    def validate(self, citations: dict):
        return SimpleNamespace(
            total_items=0,
            coverage_pct=0.0,
            type_distribution={},
            bates_gaps=[],
            bates_duplicates=[],
            line_gaps=[],
        )


def test_use_existing_copies_files_in_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("parallel_processor.PostProcessor", _DummyPostProcessor)
    monkeypatch.setattr("parallel_processor.CitationTracker", _DummyCitationTracker)

    input_pdf = tmp_path / "input" / "doc.pdf"
    input_pdf.parent.mkdir(parents=True, exist_ok=True)
    input_pdf.write_bytes(b"%PDF-1.4\n")

    use_existing = tmp_path / "existing"
    use_existing.mkdir(parents=True, exist_ok=True)
    (use_existing / "doc.md").write_text("Page 1\n1  Q  Test line\n", encoding="utf-8")
    (use_existing / "doc.json").write_text(json.dumps({"texts": []}), encoding="utf-8")

    output_dir = tmp_path / "out"
    result = process_single_document(
        pdf_path=input_pdf,
        output_dir=output_dir,
        normalized="doc",
        doc_type=DocumentType.UNKNOWN,
        is_text_based=False,
        conversion_timeout=5,
        cleanup_json=False,
        use_existing=use_existing,
    )

    assert result["status"] == "OK"
    assert (output_dir / "converted" / "doc.md").exists()
    assert (output_dir / "converted" / "doc.json").exists()


def test_pymupdf_zero_citations_falls_back_to_docling(tmp_path, monkeypatch):
    monkeypatch.setattr("parallel_processor.PostProcessor", _DummyPostProcessor)
    monkeypatch.setattr("parallel_processor.is_text_based_pdf", lambda _path: True)
    monkeypatch.setattr(
        "parallel_processor.extract_deposition",
        lambda _pdf, _out: {
            "md_path": str(tmp_path / "unused.md"),
            "citation_count": 0,
            "line_count": 0,
        },
    )

    calls = {"convert": 0}

    class _DummyDoclingConverter:
        def __init__(self, timeout=300):
            self.timeout = timeout

        def convert_document(self, input_path: str, output_dir: str):
            calls["convert"] += 1
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(input_path).stem
            md = out / f"{stem}.md"
            md.write_text("fallback content", encoding="utf-8")
            return SimpleNamespace(md_path=str(md), errors=[], citations_found={})

    monkeypatch.setattr("parallel_processor.DoclingConverter", _DummyDoclingConverter)

    input_pdf = tmp_path / "input" / "hearing_doc.pdf"
    input_pdf.parent.mkdir(parents=True, exist_ok=True)
    input_pdf.write_bytes(b"%PDF-1.4\n")

    result = process_single_document(
        pdf_path=input_pdf,
        output_dir=tmp_path / "out",
        normalized="hearing_doc",
        doc_type=DocumentType.HEARING_TRANSCRIPT,
        is_text_based=True,
        conversion_timeout=5,
        cleanup_json=False,
        use_existing=None,
    )

    assert result["status"] == "OK"
    assert result["extraction_method"] == "docling"
    assert calls["convert"] == 1


def test_pymupdf_fast_path_preserves_hearing_transcript_type(tmp_path, monkeypatch):
    monkeypatch.setattr("parallel_processor.is_text_based_pdf", lambda _path: True)
    monkeypatch.setattr(
        "parallel_processor.extract_deposition",
        lambda _pdf, _out: {
            "md_path": str(tmp_path / "out" / "converted" / "hearing_doc.md"),
            "citation_count": 5,
            "line_count": 5,
        },
    )

    input_pdf = tmp_path / "input" / "hearing_doc.pdf"
    input_pdf.parent.mkdir(parents=True, exist_ok=True)
    input_pdf.write_bytes(b"%PDF-1.4\n")

    result = process_single_document(
        pdf_path=input_pdf,
        output_dir=tmp_path / "out",
        normalized="hearing_doc",
        doc_type=DocumentType.HEARING_TRANSCRIPT,
        is_text_based=True,
        conversion_timeout=5,
        cleanup_json=False,
        use_existing=None,
    )

    assert result["status"] == "OK"
    assert result["extraction_method"] == "pymupdf"
    assert result["doc_type"] == DocumentType.HEARING_TRANSCRIPT.value
