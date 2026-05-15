"""
Hybrid retrieval combining BM25 keyword search and semantic vector search.

This module provides unified search interface using Reciprocal Rank Fusion (RRF)
to combine results from BM25 and semantic search.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from tqdm import tqdm

from lit_pipeline.citation_types import Chunk, SearchResult, DocumentType


def _should_disable_tqdm():
    """Check if progress bars should be disabled."""
    return (os.environ.get('TQDM_DISABLE', '0') == '1' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            os.environ.get('CI', '').lower() == 'true')


# Delay imports to avoid chromadb import issues at module load time
def _import_indexers():
    """Lazy import of indexers."""
    from lit_pipeline.bm25_indexer import BM25Indexer
    from lit_pipeline.vector_indexer import VectorIndexer
    return BM25Indexer, VectorIndexer

logger = logging.getLogger(__name__)


PHRASE_BOOST_FACTOR = 5.0
PHRASE_BOOST_MIN_WORDS = 4
PHRASE_BOOST_CANDIDATE_FLOOR = 50


class HybridRetriever:
    """
    Hybrid search combining BM25 and semantic vector search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both
    keyword and semantic search for better overall results.
    """

    def __init__(
        self,
        index_dir: str,
        chunks_dir: str,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        candidate_multiplier: int = 2,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.

        Args:
            index_dir: Directory containing index files
            chunks_dir: Directory containing chunk JSON files
            bm25_weight: Weight for BM25 scores (unused with RRF)
            semantic_weight: Weight for semantic scores (unused with RRF)
            candidate_multiplier: Fetch this many candidates from each index
            rrf_k: RRF parameter (default 60 is standard)
        """
        self.index_dir = Path(index_dir)
        self.chunks_dir = Path(chunks_dir)
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k

        # Initialize indexers (lazy import)
        BM25Indexer, VectorIndexer = _import_indexers()
        self.bm25_indexer = BM25Indexer(index_dir=str(self.index_dir))
        self.vector_indexer = VectorIndexer(
            persist_directory=str(self.index_dir / "chroma_db")
        )

        # Load chunk registry
        self.chunks: Dict[str, Chunk] = {}
        self._load_chunks()

        # Load indexes
        self._load_indexes()

    def _load_chunks(self) -> None:
        """Load all chunks into memory for fast lookup."""
        logger.info(f"Loading chunks from {self.chunks_dir}...")

        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))

        if not chunk_files:
            logger.warning(f"No chunk files found in {self.chunks_dir}")
            return

        chunk_count = 0
        disable_progress = _should_disable_tqdm()
        for chunk_file in tqdm(chunk_files, desc="Loading chunks", unit="file", disable=disable_progress):
            try:
                with open(chunk_file, 'r') as f:
                    chunks_data = json.load(f)

                for chunk_data in chunks_data:
                    # Reconstruct Chunk object
                    chunk = Chunk(
                        chunk_id=chunk_data["chunk_id"],
                        core_text=chunk_data["core_text"],
                        pages=chunk_data["pages"],
                        citation=chunk_data["citation"],
                        citation_string=chunk_data["citation_string"],
                        key_quotes=chunk_data.get("key_quotes", []),
                        tokens=chunk_data.get("tokens", 0),
                        doc_type=DocumentType(chunk_data.get("doc_type", "unknown")),
                        summary=chunk_data.get("summary"),
                        category=chunk_data.get("category"),
                        relevance_score=chunk_data.get("relevance_score"),
                        claims_addressed=chunk_data.get("claims_addressed"),
                        classification_method=chunk_data.get("classification_method"),
                        llm_backend=chunk_data.get("llm_backend"),
                        source_path=chunk_data.get("source_path"),
                    )
                    self.chunks[chunk.chunk_id] = chunk
                    chunk_count += 1

            except Exception as e:
                logger.error(f"Error loading {chunk_file}: {e}")

        logger.info(f"Loaded {chunk_count} chunks from {len(chunk_files)} files")

        # Save chunk registry for fast loading
        self._save_chunk_registry()

    def _save_chunk_registry(self) -> None:
        """Save chunk registry to disk atomically (write to temp, then rename)."""
        registry_path = self.index_dir / "chunk_registry.pkl"

        temp_path = registry_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        temp_path.replace(registry_path)

        logger.info(f"Chunk registry saved: {len(self.chunks)} chunks")

    def _load_chunk_registry(self) -> bool:
        """Load chunk registry from disk."""
        registry_path = self.index_dir / "chunk_registry.pkl"

        if not registry_path.exists():
            return False

        try:
            with open(registry_path, 'rb') as f:
                self.chunks = pickle.load(f)

            logger.info(f"Chunk registry loaded: {len(self.chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error loading chunk registry: {e}")
            return False

    def _load_indexes(self) -> None:
        """Load BM25 and vector indexes."""
        # Load BM25 index
        if self.bm25_indexer.is_available():
            try:
                self.bm25_indexer.load_index()
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                raise
        else:
            logger.warning("BM25 index not found")

        # Load vector index if available
        if self.vector_indexer.is_available():
            try:
                self.vector_indexer.load_index()
            except Exception as e:
                logger.warning(f"Failed to load vector index: {e}")
        else:
            logger.info("Vector search not available (Ollama not running)")

    def _apply_phrase_boost(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Boost scores for chunks that contain the query as an exact substring.

        For queries that look like phrases (>= PHRASE_BOOST_MIN_WORDS words),
        any chunk whose text contains the query verbatim gets its score
        multiplied by PHRASE_BOOST_FACTOR, then results are re-sorted.
        """
        import re

        # Only boost for phrase-like queries
        words = query.strip().split()
        if len(words) < PHRASE_BOOST_MIN_WORDS:
            return results

        # Normalize: lowercase, collapse whitespace
        normalized_query = re.sub(r'\s+', ' ', query.strip().lower())

        boosted = 0
        for result in results:
            text = result.chunk.core_text.lower()
            text = re.sub(r'\s+', ' ', text)
            if normalized_query in text:
                result.score *= PHRASE_BOOST_FACTOR
                if result.bm25_score is not None:
                    result.bm25_score *= PHRASE_BOOST_FACTOR
                boosted += 1

        if boosted:
            logger.info("Phrase boost: %d/%d results matched exact query substring",
                        boosted, len(results))
            results.sort(key=lambda r: r.score, reverse=True)
            # Re-assign ranks
            for i, result in enumerate(results):
                result.rank = i + 1

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        rerank: bool = False,
        rerank_top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query string
            top_k: Number of results to return
            mode: Search mode - "bm25", "semantic", or "hybrid"
            rerank: Whether to apply cross-encoder reranking
            rerank_top_k: Number of results after reranking (defaults to top_k)

        Returns:
            List of SearchResult objects, sorted by score descending
        """
        if not query.strip():
            return []

        # Validate mode
        if mode not in ["bm25", "semantic", "hybrid"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'bm25', 'semantic', or 'hybrid'")

        # Over-fetch candidates when query looks like a phrase or reranking
        query_words = query.strip().split()
        is_phrase = len(query_words) >= PHRASE_BOOST_MIN_WORDS

        fetch_k = top_k
        if rerank:
            fetch_k = top_k * 3
        if is_phrase:
            fetch_k = max(fetch_k, PHRASE_BOOST_CANDIDATE_FLOOR)

        # Execute searches based on mode
        if mode == "bm25":
            results = self._search_bm25(query, fetch_k)
        elif mode == "semantic":
            results = self._search_semantic(query, fetch_k)
        else:  # hybrid
            results = self._search_hybrid(query, fetch_k)

        # Apply phrase boost before reranking/trimming
        if is_phrase:
            results = self._apply_phrase_boost(query, results)

        # Apply cross-encoder reranking if requested
        if rerank and results:
            final_k = rerank_top_k if rerank_top_k is not None else top_k
            try:
                from lit_pipeline.reranker import Reranker
                reranker = Reranker()
                if reranker.is_available():
                    results = reranker.rerank(query, results, top_k=final_k)
                else:
                    logger.warning("Reranker not available, returning results without reranking")
                    results = results[:final_k]
            except ImportError:
                logger.warning("reranker module not found, returning results without reranking")
                results = results[:final_k]
        else:
            results = results[:top_k]

        return results

    def _search_bm25(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25-only search."""
        bm25_results = self.bm25_indexer.search(query, top_k=top_k)

        search_results = []
        for rank, (chunk_id, score) in enumerate(bm25_results, 1):
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                logger.warning(f"Chunk not found: {chunk_id}")
                continue

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    bm25_score=score,
                    rank=rank
                )
            )

        return search_results

    def _search_semantic(self, query: str, top_k: int) -> List[SearchResult]:
        """Semantic-only search."""
        if not self.vector_indexer.is_available():
            logger.warning("Semantic search not available, falling back to BM25")
            return self._search_bm25(query, top_k)

        semantic_results = self.vector_indexer.search(query, top_k=top_k)

        search_results = []
        for rank, (chunk_id, score) in enumerate(semantic_results, 1):
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                logger.warning(f"Chunk not found: {chunk_id}")
                continue

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=score,
                    semantic_score=score,
                    rank=rank
                )
            )

        return search_results

    def _search_hybrid(self, query: str, top_k: int) -> List[SearchResult]:
        """Hybrid search using RRF fusion."""
        # Fetch more candidates from each index
        candidate_k = top_k * self.candidate_multiplier

        # Execute both searches
        bm25_results = self.bm25_indexer.search(query, top_k=candidate_k)

        # Check if semantic search is available
        if not self.vector_indexer.is_available():
            logger.info("Semantic search not available, using BM25 only")
            return self._search_bm25(query, top_k)

        semantic_results = self.vector_indexer.search(query, top_k=candidate_k)

        # Fuse scores using RRF
        fused_results = self._fuse_scores_rrf(bm25_results, semantic_results)

        # Take top-k and create SearchResult objects
        search_results = []
        for rank, (chunk_id, rrf_score) in enumerate(fused_results[:top_k], 1):
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                logger.warning(f"Chunk not found: {chunk_id}")
                continue

            # Get individual scores
            bm25_score = next((s for cid, s in bm25_results if cid == chunk_id), None)
            semantic_score = next((s for cid, s in semantic_results if cid == chunk_id), None)

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=rrf_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    rank=rank
                )
            )

        return search_results

    def _fuse_scores_rrf(
        self,
        bm25_results: List[Tuple[str, float]],
        semantic_results: List[Tuple[str, float]],
        k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse scores using Reciprocal Rank Fusion (RRF).

        RRF formula: score(chunk) = Σ [1 / (k + rank_i)]

        Args:
            bm25_results: BM25 results as (chunk_id, score) tuples
            semantic_results: Semantic results as (chunk_id, score) tuples
            k: RRF parameter (default 60)

        Returns:
            Fused results sorted by RRF score descending
        """
        if k is None:
            k = self.rrf_k

        rrf_scores = defaultdict(float)

        # Add BM25 scores
        for rank, (chunk_id, _) in enumerate(bm25_results, 1):
            rrf_scores[chunk_id] += 1.0 / (k + rank)

        # Add semantic scores
        for rank, (chunk_id, _) in enumerate(semantic_results, 1):
            rrf_scores[chunk_id] += 1.0 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results

    def get_stats(self) -> Dict[str, any]:
        """Get index statistics."""
        stats = {
            "total_chunks": len(self.chunks),
            "bm25_available": self.bm25_indexer.is_available(),
            "semantic_available": self.vector_indexer.is_available(),
            "documents": {}
        }

        # Count chunks per document
        doc_counts = defaultdict(int)
        for chunk in self.chunks.values():
            # Extract document name from chunk_id (format: docname_chunk_####)
            doc_name = "_".join(chunk.chunk_id.split("_")[:-2])
            doc_counts[doc_name] += 1

        stats["documents"] = dict(doc_counts)
        stats["num_documents"] = len(doc_counts)

        return stats
