"""
Cross-encoder reranker for improving search precision.

Uses a cross-encoder model to score each (query, document) pair directly,
producing more accurate relevance judgments than independent embeddings.

NOTE: sentence-transformers import is delayed to runtime to gracefully
degrade when not installed.
"""

import logging
from typing import List, Optional

from lit_pipeline.citation_types import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Scores each (query, document) pair directly for more accurate
    relevance ranking than embedding similarity alone.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        self.model_name = model_name
        self._model = None
        self._error: Optional[str] = None

    def _load_model(self) -> bool:
        """
        Lazy-load the cross-encoder model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model is not None:
            return True
        if self._error is not None:
            return False

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
            return True
        except ImportError:
            self._error = "sentence-transformers not installed"
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            self._error = str(e)
            logger.warning(f"Failed to load cross-encoder model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the reranker model is available."""
        return self._load_model()

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked results with updated scores, truncated to top_k
        """
        if not results:
            return []

        if not self._load_model():
            logger.warning("Reranker not available, returning results as-is")
            return results[:top_k]

        # Build (query, document) pairs
        pairs = [(query, r.chunk.core_text) for r in results]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Assign reranker scores
        for result, score in zip(results, scores):
            result.reranker_score = float(score)

        # Sort by reranker score descending
        results.sort(key=lambda r: r.reranker_score, reverse=True)

        # Truncate and re-number ranks
        top_results = results[:top_k]
        for i, result in enumerate(top_results, 1):
            result.score = result.reranker_score
            result.rank = i

        return top_results
