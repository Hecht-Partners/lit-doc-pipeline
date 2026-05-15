"""
Tests for BM25 indexer.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from lit_pipeline.bm25_indexer import BM25Indexer
from lit_pipeline.citation_types import Chunk, DocumentType


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = [
        Chunk(
            chunk_id="doc1_chunk_0001",
            core_text="TWT technology improves battery life in WiFi devices.",
            pages=[1],
            citation={"pdf_pages": [1]},
            citation_string="Doc1, p. 1",
            doc_type=DocumentType.EXPERT_REPORT
        ),
        Chunk(
            chunk_id="doc1_chunk_0002",
            core_text="The designated corporate representative testified about TWT.",
            pages=[2],
            citation={"pdf_pages": [2]},
            citation_string="Doc1, p. 2",
            doc_type=DocumentType.DEPOSITION
        ),
        Chunk(
            chunk_id="doc2_chunk_0001",
            core_text="Patent claims describe improved power management.",
            pages=[5],
            citation={"pdf_pages": [5]},
            citation_string="Doc2, p. 5",
            doc_type=DocumentType.PATENT
        ),
    ]
    return chunks


def test_build_index(temp_index_dir, sample_chunks):
    """Test building BM25 index."""
    indexer = BM25Indexer(index_dir=temp_index_dir)
    indexer.build_index(sample_chunks)

    # Check index file exists
    index_path = Path(temp_index_dir) / "bm25_index.pkl"
    assert index_path.exists()

    # Check index attributes
    assert indexer.tfidf_matrix is not None
    assert len(indexer.chunk_ids) == 3


def test_search_keyword(temp_index_dir, sample_chunks):
    """Test keyword search."""
    indexer = BM25Indexer(index_dir=temp_index_dir)
    indexer.build_index(sample_chunks)

    # Search for "TWT"
    results = indexer.search("TWT technology", top_k=2)

    assert len(results) > 0
    # First result should contain "TWT"
    assert results[0][0] in ["doc1_chunk_0001", "doc1_chunk_0002"]
    # Scores should be normalized to [0, 1]
    assert 0 <= results[0][1] <= 1


def test_search_empty_query(temp_index_dir, sample_chunks):
    """Test search with empty query."""
    indexer = BM25Indexer(index_dir=temp_index_dir)
    indexer.build_index(sample_chunks)

    results = indexer.search("", top_k=5)
    assert len(results) == 0


def test_load_index(temp_index_dir, sample_chunks):
    """Test loading saved index."""
    # Build and save index
    indexer1 = BM25Indexer(index_dir=temp_index_dir)
    indexer1.build_index(sample_chunks)

    # Load index in new instance
    indexer2 = BM25Indexer(index_dir=temp_index_dir)
    indexer2.load_index()

    # Should have same data
    assert len(indexer2.chunk_ids) == len(indexer1.chunk_ids)
    assert indexer2.tfidf_matrix.shape == indexer1.tfidf_matrix.shape

    # Should return same results
    results1 = indexer1.search("battery life", top_k=3)
    results2 = indexer2.search("battery life", top_k=3)
    assert results1 == results2


def test_is_available(temp_index_dir, sample_chunks):
    """Test index availability check."""
    indexer = BM25Indexer(index_dir=temp_index_dir)

    # Not available before building
    assert not indexer.is_available()

    # Available after building
    indexer.build_index(sample_chunks)
    assert indexer.is_available()


def test_score_normalization(temp_index_dir, sample_chunks):
    """Test that scores are normalized to [0, 1]."""
    indexer = BM25Indexer(index_dir=temp_index_dir)
    indexer.build_index(sample_chunks)

    results = indexer.search("TWT technology battery", top_k=3)

    for chunk_id, score in results:
        assert 0 <= score <= 1

    # Top result should have score of 1.0 (after normalization)
    if results:
        assert results[0][1] == 1.0
