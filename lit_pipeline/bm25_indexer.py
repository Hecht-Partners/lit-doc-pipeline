"""
BM25 keyword search indexer using scikit-learn TfidfVectorizer.

This module provides BM25-style keyword search for litigation document chunks.
It uses TF-IDF with BM25-like parameters for efficient keyword matching.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lit_pipeline.citation_types import Chunk

logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    BM25-style keyword search using TF-IDF vectorizer.

    This provides fast keyword-based retrieval for finding chunks
    that contain specific terms or phrases.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize BM25 indexer.

        Args:
            index_dir: Directory to store index files
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter
            max_features: Maximum vocabulary size
            ngram_range: Range of n-grams (1,2) = unigrams + bigrams
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # BM25 parameters
        self.k1 = k1
        self.b = b

        # TF-IDF vectorizer (approximates BM25)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',
            norm='l2',  # L2 normalization
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Use log(tf) for BM25-like behavior
        )

        self.tfidf_matrix = None
        self.chunk_ids = []

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        # Extract text and IDs
        texts = [chunk.core_text for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Fit vectorizer and transform texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Save index
        self._save_index()

        logger.info(f"BM25 index built: {self.tfidf_matrix.shape[0]} docs, "
                   f"{self.tfidf_matrix.shape[1]} features")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() or load_index() first")

        if not query.strip():
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity (dot product for normalized vectors)
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Filter out zero scores and normalize
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                results.append((self.chunk_ids[idx], float(score)))

        # Normalize scores to [0, 1]
        if results:
            results = self._normalize_scores(results)

        return results

    def _normalize_scores(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            results: List of (chunk_id, score) tuples

        Returns:
            List with normalized scores
        """
        if not results:
            return results

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return [(chunk_id, 1.0) for chunk_id, _ in results]

        normalized = [
            (chunk_id, (score - min_score) / (max_score - min_score))
            for chunk_id, score in results
        ]

        return normalized

    def _save_index(self) -> None:
        """Save index to disk atomically (write to temp, then rename)."""
        index_path = self.index_dir / "bm25_index.pkl"

        index_data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "chunk_ids": self.chunk_ids,
            "k1": self.k1,
            "b": self.b
        }

        temp_path = index_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(index_data, f)
        temp_path.replace(index_path)

        logger.info(f"BM25 index saved to {index_path}")

    def load_index(self) -> None:
        """Load index from disk."""
        index_path = self.index_dir / "bm25_index.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)

        self.vectorizer = index_data["vectorizer"]
        self.tfidf_matrix = index_data["tfidf_matrix"]
        self.chunk_ids = index_data["chunk_ids"]
        self.k1 = index_data.get("k1", self.k1)
        self.b = index_data.get("b", self.b)

        logger.info(f"BM25 index loaded: {len(self.chunk_ids)} chunks")

    def is_available(self) -> bool:
        """Check if index is available."""
        index_path = self.index_dir / "bm25_index.pkl"
        return index_path.exists()
