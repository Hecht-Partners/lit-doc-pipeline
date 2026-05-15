"""
Index state management for incremental indexing.

Tracks which documents have been indexed and their content hashes,
enabling efficient incremental updates without full reindexing.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocument:
    """State tracking for an indexed document."""
    filename: str
    file_path: str
    content_hash: str  # SHA256 of file content
    last_indexed: str  # ISO timestamp
    chunk_count: int
    index_types: List[str]  # ["bm25", "vector", "enriched"]

    def needs_reindex(self, current_hash: str) -> bool:
        """Check if document needs reindexing based on content hash."""
        return self.content_hash != current_hash


class IndexState:
    """
    Manages index state for incremental indexing.

    State file format:
    {
        "version": "1.0",
        "last_updated": "2026-02-12T12:00:00",
        "bm25_index_path": "indexes/bm25_index.pkl",
        "vector_index_path": "indexes/chroma_db",
        "documents": {
            "daniel_alexander_10_24_2025_chunks.json": {
                "filename": "daniel_alexander_10_24_2025_chunks.json",
                "file_path": "converted/daniel_alexander_10_24_2025_chunks.json",
                "content_hash": "abc123...",
                "last_indexed": "2026-02-12T12:00:00",
                "chunk_count": 42,
                "index_types": ["bm25", "vector"]
            }
        }
    }
    """

    STATE_FILE = ".lit-index-state.json"
    VERSION = "1.0"

    def __init__(self, index_dir: Path):
        """
        Initialize index state.

        Args:
            index_dir: Directory where indexes and state file are stored
        """
        self.index_dir = index_dir
        self.state_file = index_dir / self.STATE_FILE
        self.documents: Dict[str, IndexedDocument] = {}
        self.last_updated: Optional[str] = None
        self.bm25_index_path: Optional[str] = None
        self.vector_index_path: Optional[str] = None

        # Load existing state if present
        if self.state_file.exists():
            self.load()
        else:
            self.last_updated = datetime.now().isoformat()

    def load(self):
        """Load state from file."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            self.last_updated = data.get("last_updated")
            self.bm25_index_path = data.get("bm25_index_path")
            self.vector_index_path = data.get("vector_index_path")

            # Reconstruct IndexedDocument objects
            self.documents = {}
            for filename, doc_data in data.get("documents", {}).items():
                self.documents[filename] = IndexedDocument(**doc_data)

            logger.info("Loaded index state from %s (%d documents)",
                       self.state_file, len(self.documents))

        except Exception as e:
            logger.warning("Failed to load index state file: %s", e)
            self.last_updated = datetime.now().isoformat()

    def save(self):
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()

        data = {
            "version": self.VERSION,
            "last_updated": self.last_updated,
            "bm25_index_path": self.bm25_index_path,
            "vector_index_path": self.vector_index_path,
            "documents": {
                filename: asdict(doc) for filename, doc in self.documents.items()
            }
        }

        try:
            # Ensure directory exists
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Write atomically (write to temp, then rename)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)

            logger.debug("Saved index state to %s", self.state_file)

        except Exception as e:
            logger.error("Failed to save index state file: %s", e)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """
        Compute SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read in 64KB chunks for memory efficiency
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_document(self, filename: str) -> Optional[IndexedDocument]:
        """
        Get indexed document state.

        Args:
            filename: Document filename

        Returns:
            IndexedDocument if found, None otherwise
        """
        return self.documents.get(filename)

    def add_document(
        self,
        filename: str,
        file_path: str,
        content_hash: str,
        chunk_count: int,
        index_types: List[str],
    ):
        """
        Add or update document in index state.

        Args:
            filename: Document filename
            file_path: Relative path to document
            content_hash: SHA256 hash of content
            chunk_count: Number of chunks in document
            index_types: Types of indexes containing this document
        """
        self.documents[filename] = IndexedDocument(
            filename=filename,
            file_path=file_path,
            content_hash=content_hash,
            last_indexed=datetime.now().isoformat(),
            chunk_count=chunk_count,
            index_types=index_types,
        )

    def needs_reindex(self, file_path: Path, force: bool = False) -> bool:
        """
        Check if document needs reindexing.

        Args:
            file_path: Path to document file
            force: Force reindexing even if content unchanged

        Returns:
            True if should reindex, False to skip
        """
        if force:
            return True

        filename = file_path.name

        # If not in index, needs indexing
        if filename not in self.documents:
            return True

        # If file doesn't exist, skip
        if not file_path.exists():
            logger.warning("File not found: %s", file_path)
            return False

        # Check if content changed
        current_hash = self.compute_file_hash(file_path)
        indexed_doc = self.documents[filename]

        if indexed_doc.needs_reindex(current_hash):
            logger.debug("Document changed: %s (hash mismatch)", filename)
            return True

        return False

    def get_documents_to_reindex(
        self,
        chunks_dir: Path,
        force: bool = False,
    ) -> List[Path]:
        """
        Get list of documents that need reindexing.

        Args:
            chunks_dir: Directory containing chunk files
            force: Force reindex all documents

        Returns:
            List of chunk file paths to reindex
        """
        if force:
            logger.info("Force reindex: processing all documents")
            return sorted(chunks_dir.glob("*_chunks.json"))

        all_chunks = sorted(chunks_dir.glob("*_chunks.json"))
        to_reindex = []

        for chunk_file in all_chunks:
            if self.needs_reindex(chunk_file, force):
                to_reindex.append(chunk_file)

        if not to_reindex:
            logger.info("All documents up to date, no reindexing needed")
        else:
            logger.info("Reindexing %d/%d documents (changed or new)",
                       len(to_reindex), len(all_chunks))

        return to_reindex

    def remove_document(self, filename: str):
        """
        Remove document from index state.

        Args:
            filename: Document filename to remove
        """
        if filename in self.documents:
            del self.documents[filename]
            logger.debug("Removed document from index state: %s", filename)

    def prune_missing_files(self, chunks_dir: Path):
        """
        Remove documents from state that no longer exist.

        Args:
            chunks_dir: Directory containing chunk files
        """
        to_remove = []

        for filename, doc in self.documents.items():
            file_path = chunks_dir / filename
            if not file_path.exists():
                logger.info("Pruning missing file: %s", filename)
                to_remove.append(filename)

        for filename in to_remove:
            self.remove_document(filename)

        if to_remove:
            self.save()

    def summary(self) -> str:
        """Generate summary of index state."""
        total = len(self.documents)
        if total == 0:
            return "No documents indexed yet"

        total_chunks = sum(doc.chunk_count for doc in self.documents.values())

        # Count by index type
        bm25_count = sum(1 for doc in self.documents.values() if "bm25" in doc.index_types)
        vector_count = sum(1 for doc in self.documents.values() if "vector" in doc.index_types)

        lines = [
            f"Index State Summary:",
            f"  Total documents: {total}",
            f"  Total chunks: {total_chunks}",
            f"  BM25 indexed: {bm25_count}",
            f"  Vector indexed: {vector_count}",
            f"  Last updated: {self.last_updated}",
        ]

        if self.bm25_index_path:
            lines.append(f"  BM25 index: {self.bm25_index_path}")
        if self.vector_index_path:
            lines.append(f"  Vector index: {self.vector_index_path}")

        return "\n".join(lines)
