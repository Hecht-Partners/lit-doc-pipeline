import json
from pathlib import Path

from lit_pipeline.index_state import IndexState
from lit_pipeline.lit_doc_retriever import build_indexes
from lit_pipeline.post_processor import PostProcessor


def test_build_indexes_tracks_chunk_count_per_document(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "chunk_id": "doc1_chunk_0000",
            "core_text": "alpha",
            "pages": [1],
            "citation": {},
            "citation_string": "Doc1 p.1",
            "doc_type": "unknown",
        },
        {
            "chunk_id": "doc1_chunk_0001",
            "core_text": "beta",
            "pages": [1],
            "citation": {},
            "citation_string": "Doc1 p.1",
            "doc_type": "unknown",
        },
    ]
    with open(converted_dir / "doc1_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    class _DummyBM25Indexer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_index(self, chunks):
            self.chunks = chunks

    class _DummyVectorIndexer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def is_available(self):
            return False

        def build_index(self, *args, **kwargs):
            raise AssertionError("Vector index should not be built in this test")

    monkeypatch.setattr(
        "lit_pipeline.lit_doc_retriever._import_indexers",
        lambda: (_DummyBM25Indexer, _DummyVectorIndexer, None),
    )

    build_indexes(str(output_dir))

    state = IndexState(output_dir / "indexes")
    indexed = state.get_document("doc1_chunks.json")
    assert indexed is not None
    assert indexed.chunk_count == 2


def test_clean_markdown_uses_current_line_index_for_concordance_detection(monkeypatch):
    processor = PostProcessor()
    seen_indices = []

    def _capture_indices(lines, start_idx):
        seen_indices.append(start_idx)
        return False

    monkeypatch.setattr(processor, "_is_concordance_section", _capture_indices)

    content = "\n".join([
        "Intro text",
        "A",
        "Middle",
        "A",
        "End",
    ])

    processor._clean_markdown(content)

    assert seen_indices == [1, 3]
