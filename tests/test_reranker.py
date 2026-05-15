"""
Tests for cross-encoder reranker.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from lit_pipeline.citation_types import Chunk, SearchResult, DocumentType
from lit_pipeline.reranker import Reranker


def _make_result(chunk_id: str, text: str, score: float, rank: int) -> SearchResult:
    """Helper to create a SearchResult for testing."""
    chunk = Chunk(
        chunk_id=chunk_id,
        core_text=text,
        pages=[1],
        citation={"pdf_pages": [1]},
        citation_string=f"{chunk_id}, p. 1",
        doc_type=DocumentType.UNKNOWN,
    )
    return SearchResult(
        chunk_id=chunk_id,
        chunk=chunk,
        score=score,
        bm25_score=score,
        rank=rank,
    )


def test_reranker_not_available():
    """Verify is_available() returns False when sentence-transformers is missing."""
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        reranker = Reranker()
        # Force fresh state
        reranker._model = None
        reranker._error = None
        assert reranker.is_available() is False


def test_rerank_empty_results():
    """Verify empty input returns empty output."""
    reranker = Reranker()
    result = reranker.rerank("some query", [], top_k=5)
    assert result == []


def test_rerank_reorders_results():
    """With a mock model, verify reranker re-sorts results and assigns reranker_score."""
    reranker = Reranker()

    # Create results where BM25 order differs from reranker order
    results = [
        _make_result("chunk_a", "not very relevant text", 0.9, 1),
        _make_result("chunk_b", "somewhat relevant text", 0.7, 2),
        _make_result("chunk_c", "highly relevant target text", 0.5, 3),
    ]

    # Mock the model to return scores that reverse the original order
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.5, 0.9]  # chunk_c gets highest
    reranker._model = mock_model

    reranked = reranker.rerank("target text", results, top_k=3)

    assert len(reranked) == 3
    # chunk_c should now be first (highest reranker score)
    assert reranked[0].chunk_id == "chunk_c"
    assert reranked[0].reranker_score == pytest.approx(0.9)
    assert reranked[0].rank == 1
    # chunk_b second
    assert reranked[1].chunk_id == "chunk_b"
    assert reranked[1].reranker_score == pytest.approx(0.5)
    assert reranked[1].rank == 2
    # chunk_a last
    assert reranked[2].chunk_id == "chunk_a"
    assert reranked[2].reranker_score == pytest.approx(0.1)
    assert reranked[2].rank == 3


def test_rerank_respects_top_k():
    """Feed 10 results, request top-3, verify only 3 returned."""
    reranker = Reranker()

    results = [
        _make_result(f"chunk_{i}", f"text number {i}", 1.0 - i * 0.1, i + 1)
        for i in range(10)
    ]

    # Mock model returns descending scores
    mock_model = MagicMock()
    mock_model.predict.return_value = [float(10 - i) for i in range(10)]
    reranker._model = mock_model

    reranked = reranker.rerank("query", results, top_k=3)

    assert len(reranked) == 3
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2
    assert reranked[2].rank == 3


def test_rerank_fallback_when_unavailable():
    """When reranker can't load, results are returned truncated but unmodified."""
    reranker = Reranker()
    reranker._error = "not installed"  # Simulate failed load

    results = [
        _make_result("chunk_a", "text a", 0.9, 1),
        _make_result("chunk_b", "text b", 0.7, 2),
        _make_result("chunk_c", "text c", 0.5, 3),
    ]

    reranked = reranker.rerank("query", results, top_k=2)

    assert len(reranked) == 2
    # Order unchanged, just truncated
    assert reranked[0].chunk_id == "chunk_a"
    assert reranked[1].chunk_id == "chunk_b"
    # No reranker_score assigned
    assert reranked[0].reranker_score is None


def test_search_with_rerank_flag():
    """End-to-end: HybridRetriever.search(rerank=True) applies reranking."""
    from lit_pipeline.bm25_indexer import BM25Indexer
    from lit_pipeline.hybrid_retriever import HybridRetriever

    index_dir = tempfile.mkdtemp()
    chunks_dir = tempfile.mkdtemp()

    try:
        chunks_data = [
            {
                "chunk_id": "doc1_chunk_0001",
                "core_text": "TWT technology improves battery life in WiFi 6 devices.",
                "pages": [1],
                "citation": {"pdf_pages": [1]},
                "citation_string": "Doc1, p. 1",
                "key_quotes": [],
                "tokens": 12,
                "doc_type": "expert_report",
            },
            {
                "chunk_id": "doc1_chunk_0002",
                "core_text": "Corporate representative testified about WiFi certification.",
                "pages": [2],
                "citation": {"pdf_pages": [2]},
                "citation_string": "Doc1, p. 2",
                "key_quotes": [],
                "tokens": 10,
                "doc_type": "deposition",
            },
        ]

        chunks_file = Path(chunks_dir) / "doc1_chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(chunks_data, f)

        # Build BM25 index
        chunks = []
        for cd in chunks_data:
            chunks.append(
                Chunk(
                    chunk_id=cd["chunk_id"],
                    core_text=cd["core_text"],
                    pages=cd["pages"],
                    citation=cd["citation"],
                    citation_string=cd["citation_string"],
                    doc_type=DocumentType(cd["doc_type"]),
                )
            )

        bm25 = BM25Indexer(index_dir=index_dir)
        bm25.build_index(chunks)

        retriever = HybridRetriever(index_dir=index_dir, chunks_dir=chunks_dir)

        # Mock the reranker to avoid needing sentence-transformers
        mock_model = MagicMock()
        # Score the second chunk higher to prove reranking works
        mock_model.predict.return_value = [0.3, 0.9]

        with patch("lit_pipeline.reranker.Reranker._load_model") as mock_load:
            mock_load.return_value = True
            with patch("lit_pipeline.reranker.Reranker.__init__", lambda self, **kw: None):
                with patch("lit_pipeline.reranker.Reranker.is_available", return_value=True):
                    with patch("lit_pipeline.reranker.Reranker.rerank") as mock_rerank:
                        # Simulate what rerank does
                        def fake_rerank(query, results, top_k=10):
                            scores = [0.3, 0.9]
                            for r, s in zip(results, scores):
                                r.reranker_score = s
                            results.sort(
                                key=lambda r: r.reranker_score, reverse=True
                            )
                            for i, r in enumerate(results[:top_k], 1):
                                r.score = r.reranker_score
                                r.rank = i
                            return results[:top_k]

                        mock_rerank.side_effect = fake_rerank

                        results = retriever.search(
                            "WiFi", top_k=2, mode="bm25", rerank=True
                        )

                        assert len(results) > 0
                        # Reranker should have been called
                        mock_rerank.assert_called_once()

    finally:
        shutil.rmtree(index_dir, ignore_errors=True)
        shutil.rmtree(chunks_dir, ignore_errors=True)
