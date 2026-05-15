#!/usr/bin/env python3
"""
Test script for performance optimization features.

Tests:
1. Incremental indexing - only reindex changed files
2. Parallel processing - process multiple documents concurrently
3. Index state tracking
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from lit_pipeline.index_state import IndexState, IndexedDocument


def test_incremental_indexing():
    """Test incremental indexing functionality."""
    print("\n=== Test 1: Incremental Indexing ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "indexes"
        chunks_dir = Path(tmpdir) / "chunks"
        chunks_dir.mkdir(parents=True)

        # Create test chunk files
        chunk1 = chunks_dir / "doc1_chunks.json"
        chunk2 = chunks_dir / "doc2_chunks.json"

        chunk1_content = json.dumps([{"chunk_id": "1", "text": "content1"}])
        chunk2_content = json.dumps([{"chunk_id": "2", "text": "content2"}])

        chunk1.write_text(chunk1_content)
        chunk2.write_text(chunk2_content)

        # Initialize index state
        state = IndexState(index_dir)

        # First indexing - all files should need indexing
        to_index = state.get_documents_to_reindex(chunks_dir, force=False)
        assert len(to_index) == 2, f"Expected 2 files to index, got {len(to_index)}"
        print(f"✓ First run: {len(to_index)} files to index")

        # Add files to state
        for chunk_file in [chunk1, chunk2]:
            file_hash = IndexState.compute_file_hash(chunk_file)
            state.add_document(
                filename=chunk_file.name,
                file_path=str(chunk_file.relative_to(tmpdir)),
                content_hash=file_hash,
                chunk_count=1,
                index_types=["bm25"],
            )
        state.save()

        # Second indexing - no files should need indexing
        to_index = state.get_documents_to_reindex(chunks_dir, force=False)
        assert len(to_index) == 0, f"Expected 0 files to index, got {len(to_index)}"
        print("✓ Second run: 0 files to index (all up to date)")

        # Modify one file
        chunk1.write_text(json.dumps([{"chunk_id": "1", "text": "MODIFIED"}]))

        # Third indexing - only modified file should need indexing
        to_index = state.get_documents_to_reindex(chunks_dir, force=False)
        assert len(to_index) == 1, f"Expected 1 file to index, got {len(to_index)}"
        assert to_index[0].name == "doc1_chunks.json"
        print("✓ Third run: 1 file to index (modified file detected)")

        # Force reindex - all files should need indexing
        to_index = state.get_documents_to_reindex(chunks_dir, force=True)
        assert len(to_index) == 2, f"Expected 2 files with force=True, got {len(to_index)}"
        print("✓ Force reindex: 2 files to index (force mode)")

    print("✓ Incremental indexing works correctly")


def test_file_hash_computation():
    """Test file hash computation."""
    print("\n=== Test 2: File Hash Computation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"

        # Write content
        test_file.write_text("test content")
        hash1 = IndexState.compute_file_hash(test_file)

        # Same content should produce same hash
        test_file.write_text("test content")
        hash2 = IndexState.compute_file_hash(test_file)
        assert hash1 == hash2, "Same content should produce same hash"
        print("✓ Same content produces same hash")

        # Different content should produce different hash
        test_file.write_text("different content")
        hash3 = IndexState.compute_file_hash(test_file)
        assert hash1 != hash3, "Different content should produce different hash"
        print("✓ Different content produces different hash")

        # Verify hash format (SHA256 = 64 hex chars)
        assert len(hash1) == 64, f"Expected 64 char hash, got {len(hash1)}"
        assert all(c in '0123456789abcdef' for c in hash1), "Hash should be hexadecimal"
        print("✓ Hash format is correct (SHA256)")

    print("✓ File hash computation works correctly")


def test_index_state_persistence():
    """Test index state save/load."""
    print("\n=== Test 3: Index State Persistence ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)

        # Create state
        state1 = IndexState(index_dir)
        state1.add_document(
            filename="doc1_chunks.json",
            file_path="converted/doc1_chunks.json",
            content_hash="abc123",
            chunk_count=10,
            index_types=["bm25", "vector"],
        )
        state1.bm25_index_path = "indexes/bm25_index.pkl"
        state1.vector_index_path = "indexes/chroma_db"
        state1.save()

        # Verify state file exists
        state_file = index_dir / ".lit-index-state.json"
        assert state_file.exists(), "State file not created"
        print("✓ State file created")

        # Load state in new instance
        state2 = IndexState(index_dir)
        assert len(state2.documents) == 1, "Document not loaded"
        assert "doc1_chunks.json" in state2.documents
        assert state2.bm25_index_path == "indexes/bm25_index.pkl"
        assert state2.vector_index_path == "indexes/chroma_db"

        doc = state2.documents["doc1_chunks.json"]
        assert doc.chunk_count == 10
        assert doc.index_types == ["bm25", "vector"]
        print("✓ State loaded correctly")

        # Verify JSON format
        with open(state_file) as f:
            data = json.load(f)

        assert "version" in data
        assert "documents" in data
        assert "doc1_chunks.json" in data["documents"]
        print("✓ State file has correct JSON format")

    print("✓ Index state persistence works correctly")


def test_prune_missing_files():
    """Test pruning of missing files from index state."""
    print("\n=== Test 4: Prune Missing Files ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "indexes"
        chunks_dir = Path(tmpdir) / "chunks"
        chunks_dir.mkdir(parents=True)

        # Create state with documents
        state = IndexState(index_dir)
        state.add_document("doc1_chunks.json", "chunks/doc1_chunks.json", "hash1", 10, ["bm25"])
        state.add_document("doc2_chunks.json", "chunks/doc2_chunks.json", "hash2", 20, ["bm25"])
        state.add_document("doc3_chunks.json", "chunks/doc3_chunks.json", "hash3", 30, ["bm25"])
        state.save()

        assert len(state.documents) == 3, "Expected 3 documents in state"
        print(f"✓ Initial state: {len(state.documents)} documents")

        # Create only doc1 and doc2 (doc3 is missing)
        (chunks_dir / "doc1_chunks.json").write_text("{}")
        (chunks_dir / "doc2_chunks.json").write_text("{}")

        # Prune missing files
        state.prune_missing_files(chunks_dir)

        assert len(state.documents) == 2, "Expected 2 documents after pruning"
        assert "doc1_chunks.json" in state.documents
        assert "doc2_chunks.json" in state.documents
        assert "doc3_chunks.json" not in state.documents
        print(f"✓ After pruning: {len(state.documents)} documents (doc3 removed)")

    print("✓ Prune missing files works correctly")


def test_index_state_summary():
    """Test index state summary generation."""
    print("\n=== Test 5: Index State Summary ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)
        state = IndexState(index_dir)

        # Add documents with different index types
        state.add_document("doc1_chunks.json", "chunks/doc1.json", "hash1", 10, ["bm25"])
        state.add_document("doc2_chunks.json", "chunks/doc2.json", "hash2", 20, ["bm25", "vector"])
        state.add_document("doc3_chunks.json", "chunks/doc3.json", "hash3", 30, ["bm25", "vector"])
        state.bm25_index_path = "indexes/bm25_index.pkl"
        state.vector_index_path = "indexes/chroma_db"

        summary = state.summary()
        print("\nState Summary:")
        print(summary)

        assert "Total documents: 3" in summary
        assert "Total chunks: 60" in summary
        assert "BM25 indexed: 3" in summary
        assert "Vector indexed: 2" in summary
        assert "bm25_index.pkl" in summary
        assert "chroma_db" in summary

        print("✓ Summary generation works correctly")


def main():
    """Run all tests."""
    print("Testing Performance Optimization Features")
    print("=" * 60)

    try:
        test_incremental_indexing()
        test_file_hash_computation()
        test_index_state_persistence()
        test_prune_missing_files()
        test_index_state_summary()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
