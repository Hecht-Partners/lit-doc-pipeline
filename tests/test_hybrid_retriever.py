"""
Tests for hybrid retriever.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from lit_pipeline.hybrid_retriever import HybridRetriever
from lit_pipeline.bm25_indexer import BM25Indexer
from lit_pipeline.citation_types import Chunk, DocumentType


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    index_dir = tempfile.mkdtemp()
    chunks_dir = tempfile.mkdtemp()
    yield index_dir, chunks_dir
    shutil.rmtree(index_dir, ignore_errors=True)
    shutil.rmtree(chunks_dir, ignore_errors=True)


@pytest.fixture
def sample_chunks_with_files(temp_dirs):
    """Create sample chunks and save to JSON files."""
    _, chunks_dir = temp_dirs

    chunks = [
        {
            "chunk_id": "doc1_chunk_0001",
            "core_text": "TWT technology improves battery life in WiFi 6 devices by enabling target wake time.",
            "pages": [1, 2],
            "citation": {"pdf_pages": [1, 2]},
            "citation_string": "Doc1, pp. 1-2",
            "key_quotes": [],
            "tokens": 15,
            "doc_type": "expert_report"
        },
        {
            "chunk_id": "doc1_chunk_0002",
            "core_text": "The designated corporate representative testified about WiFi certification.",
            "pages": [3],
            "citation": {"pdf_pages": [3]},
            "citation_string": "Doc1, p. 3",
            "key_quotes": [],
            "tokens": 12,
            "doc_type": "deposition"
        },
        {
            "chunk_id": "doc2_chunk_0001",
            "core_text": "Patent claims describe power management improvements using low power modes.",
            "pages": [5, 6],
            "citation": {"pdf_pages": [5, 6]},
            "citation_string": "Doc2, pp. 5-6",
            "key_quotes": [],
            "tokens": 13,
            "doc_type": "patent"
        },
    ]

    # Save to file
    chunks_file = Path(chunks_dir) / "doc1_chunks.json"
    with open(chunks_file, 'w') as f:
        json.dump(chunks, f)

    return chunks


def test_load_chunks(temp_dirs, sample_chunks_with_files):
    """Test loading chunks from JSON files."""
    index_dir, chunks_dir = temp_dirs

    retriever = HybridRetriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir
    )

    # Should have loaded 3 chunks
    assert len(retriever.chunks) == 3
    assert "doc1_chunk_0001" in retriever.chunks
    assert "doc2_chunk_0001" in retriever.chunks


def test_build_and_search_bm25(temp_dirs, sample_chunks_with_files):
    """Test BM25-only search."""
    index_dir, chunks_dir = temp_dirs

    # Build index first
    chunks = []
    for chunk_data in sample_chunks_with_files:
        chunk = Chunk(
            chunk_id=chunk_data["chunk_id"],
            core_text=chunk_data["core_text"],
            pages=chunk_data["pages"],
            citation=chunk_data["citation"],
            citation_string=chunk_data["citation_string"],
            doc_type=DocumentType(chunk_data["doc_type"])
        )
        chunks.append(chunk)

    bm25_indexer = BM25Indexer(index_dir=index_dir)
    bm25_indexer.build_index(chunks)

    # Search
    retriever = HybridRetriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir
    )

    results = retriever.search("TWT battery life", top_k=2, mode="bm25")

    assert len(results) > 0
    assert results[0].chunk_id == "doc1_chunk_0001"  # Should match TWT chunk
    assert results[0].bm25_score is not None
    assert results[0].rank == 1


def test_search_modes(temp_dirs, sample_chunks_with_files):
    """Test different search modes."""
    index_dir, chunks_dir = temp_dirs

    # Build index
    chunks = []
    for chunk_data in sample_chunks_with_files:
        chunk = Chunk(
            chunk_id=chunk_data["chunk_id"],
            core_text=chunk_data["core_text"],
            pages=chunk_data["pages"],
            citation=chunk_data["citation"],
            citation_string=chunk_data["citation_string"],
            doc_type=DocumentType(chunk_data["doc_type"])
        )
        chunks.append(chunk)

    bm25_indexer = BM25Indexer(index_dir=index_dir)
    bm25_indexer.build_index(chunks)

    retriever = HybridRetriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir
    )

    # Test BM25 mode
    results_bm25 = retriever.search("power management", top_k=2, mode="bm25")
    assert len(results_bm25) > 0

    # Test hybrid mode
    # If semantic is available, it will use hybrid; otherwise falls back to BM25
    try:
        results_hybrid = retriever.search("power management", top_k=2, mode="hybrid")
        assert len(results_hybrid) > 0
    except Exception as e:
        # Expected if semantic index not built or Ollama unavailable
        pytest.skip(f"Semantic search not available: {e}")


def test_rrf_fusion(temp_dirs):
    """Test RRF score fusion algorithm."""
    index_dir, chunks_dir = temp_dirs

    retriever = HybridRetriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        rrf_k=60
    )

    # Test RRF calculation
    bm25_results = [("chunk1", 0.9), ("chunk2", 0.7), ("chunk3", 0.5)]
    semantic_results = [("chunk2", 0.8), ("chunk1", 0.6), ("chunk4", 0.4)]

    fused = retriever._fuse_scores_rrf(bm25_results, semantic_results)

    # chunk1 and chunk2 should have highest scores (in both lists)
    assert fused[0][0] in ["chunk1", "chunk2"]
    assert fused[1][0] in ["chunk1", "chunk2"]

    # RRF scores should be positive
    for chunk_id, score in fused:
        assert score > 0


def test_get_stats(temp_dirs, sample_chunks_with_files):
    """Test retrieval statistics."""
    index_dir, chunks_dir = temp_dirs

    retriever = HybridRetriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir
    )

    stats = retriever.get_stats()

    assert stats["total_chunks"] == 3
    assert stats["num_documents"] == 2  # doc1 and doc2
    assert "doc1" in stats["documents"]
    assert "doc2" in stats["documents"]
    assert stats["bm25_available"] == False  # Not built yet
    # Semantic may be available if ChromaDB + Ollama are running
    assert "semantic_available" in stats


def test_empty_query(temp_dirs, sample_chunks_with_files):
    """Test search with empty query."""
    index_dir, chunks_dir = temp_dirs

    # Build index
    chunks = []
    for chunk_data in sample_chunks_with_files:
        chunk = Chunk(
            chunk_id=chunk_data["chunk_id"],
            core_text=chunk_data["core_text"],
            pages=chunk_data["pages"],
            citation=chunk_data["citation"],
            citation_string=chunk_data["citation_string"],
            doc_type=DocumentType(chunk_data["doc_type"])
        )
        chunks.append(chunk)

    bm25_indexer = BM25Indexer(index_dir=index_dir)
    bm25_indexer.build_index(chunks)

    retriever = HybridRetriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir
    )

    results = retriever.search("", top_k=5, mode="bm25")
    assert len(results) == 0
