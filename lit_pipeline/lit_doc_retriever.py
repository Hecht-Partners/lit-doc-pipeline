#!/usr/bin/env python3
"""
CLI tool for searching litigation documents.

This tool provides command-line access to hybrid search (BM25 + semantic)
over processed litigation document chunks.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from lit_pipeline.citation_types import Chunk, DocumentType
from lit_pipeline.index_state import IndexState


def _should_disable_tqdm():
    """Check if progress bars should be disabled."""
    return (os.environ.get('TQDM_DISABLE', '0') == '1' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            os.environ.get('CI', '').lower() == 'true')


# Delay imports to avoid chromadb import issues
def _import_indexers():
    """Lazy import of indexers to avoid chromadb import-time errors."""
    from lit_pipeline.bm25_indexer import BM25Indexer
    from lit_pipeline.vector_indexer import VectorIndexer
    from lit_pipeline.hybrid_retriever import HybridRetriever
    return BM25Indexer, VectorIndexer, HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _char_pos_to_line_idx(text: str, char_pos: int) -> int:
    """Convert a character position in text to a 0-based line index."""
    return text[:char_pos].count("\n")


def _find_query_page(query: str, chunk) -> tuple:
    """
    Find the page (and bates) where the query best matches in the chunk.

    Returns:
        (page, bates) or (None, None) if no page_map available.
    """
    page_map = chunk.citation.get("page_map")
    if not page_map:
        return None, None

    bates_map = chunk.citation.get("bates_map")
    core = chunk.core_text

    # Try exact case-insensitive substring match
    lower_core = core.lower()
    lower_query = query.lower()
    pos = lower_core.find(lower_query)
    if pos >= 0:
        line_idx = _char_pos_to_line_idx(core, pos)
        line_idx = min(line_idx, len(page_map) - 1)
        page = page_map[line_idx]
        bates = bates_map[line_idx] if bates_map and line_idx < len(bates_map) else None
        return page, bates

    # Fallback: find line with most query term overlap
    query_terms = set(lower_query.split())
    if not query_terms:
        return None, None

    lines = core.split("\n")
    best_idx = 0
    best_overlap = 0
    for i, line in enumerate(lines):
        line_words = set(line.lower().split())
        overlap = len(query_terms & line_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    if best_overlap > 0 and best_idx < len(page_map):
        page = page_map[best_idx]
        bates = bates_map[best_idx] if bates_map and best_idx < len(bates_map) else None
        return page, bates

    return None, None


def build_indexes(output_dir: str, config_path: Optional[str] = None, force: bool = False) -> None:
    """
    Build BM25 and vector indexes from chunk files.

    Args:
        output_dir: Directory containing chunk JSON files
        config_path: Path to retrieval config (optional)
        force: Force rebuild even if chunks unchanged (default: False)
    """
    BM25Indexer, VectorIndexer, _ = _import_indexers()

    output_path = Path(output_dir)
    converted_dir = output_path / "converted"
    index_dir = output_path / "indexes"

    # Initialize index state for incremental indexing
    index_state = IndexState(index_dir)

    if not converted_dir.exists():
        logger.error(f"Chunks directory not found: {converted_dir}")
        sys.exit(1)

    # Load config if provided
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Prune deleted files from state
    index_state.prune_missing_files(converted_dir)

    # Determine which files need reindexing
    # Note: get_documents_to_reindex() includes logging
    files_to_reindex = index_state.get_documents_to_reindex(converted_dir, force=force)

    # Early exit if nothing to do
    if not files_to_reindex:
        logger.info("Use --force-rebuild to force reindexing")
        return

    # Extract document stems from changed files for vector re-embedding
    changed_stems = set()
    for chunk_file in files_to_reindex:
        # chunk_file.stem is like "daniel_alexander_10_24_2025_chunks"
        # Remove "_chunks" suffix to get document stem
        stem = chunk_file.stem.replace("_chunks", "")
        changed_stems.add(stem)

    logger.info(f"Changed documents: {len(changed_stems)} ({len(files_to_reindex)} chunk files)")

    # Load ALL chunks (BM25 requires full corpus for TF-IDF; vector indexer will use incrementally)
    logger.info("Loading chunks from cache...")
    chunks = []
    chunk_files = sorted(converted_dir.glob("*_chunks.json"))

    if not chunk_files:
        logger.error(f"No chunk files found in {converted_dir}")
        sys.exit(1)

    disable_progress = _should_disable_tqdm()
    for chunk_file in tqdm(chunk_files, desc="Loading chunks", unit="file", disable=disable_progress):
        try:
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)

            for chunk_data in chunks_data:
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
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {e}")
            continue

    logger.info(f"Loaded {len(chunks)} chunks from {len(chunk_files)} files")

    # Build BM25 index
    logger.info("\nBuilding BM25 index...")
    bm25_config = config.get("bm25", {})
    bm25_indexer = BM25Indexer(
        index_dir=str(index_dir),
        k1=bm25_config.get("k1", 1.5),
        b=bm25_config.get("b", 0.75),
        max_features=bm25_config.get("max_features", None),
        ngram_range=tuple(bm25_config.get("ngram_range", [1, 2]))
    )

    start_time = time.time()
    bm25_indexer.build_index(chunks)
    bm25_time = time.time() - start_time
    logger.info(f"BM25 index built in {bm25_time:.2f}s")

    # Build vector index (if Ollama available)
    logger.info("\nBuilding vector index...")
    chroma_config = config.get("chroma", {})
    vector_indexer = VectorIndexer(
        persist_directory=str(index_dir / "chroma_db"),
        embedding_model=chroma_config.get("embedding_model", "nomic-embed-text"),
        ollama_url=chroma_config.get("ollama_url", "http://localhost:11434/api/embeddings")
    )

    if vector_indexer.is_available():
        start_time = time.time()

        # Auto-detect incremental unless forced
        incremental_override = False if force else None

        # Build/update index
        vector_indexer.build_index(
            chunks,
            incremental=incremental_override,
            changed_stems=changed_stems if changed_stems else None
        )

        vector_time = time.time() - start_time
        logger.info(f"Vector index built in {vector_time:.2f}s")
    else:
        logger.warning("Ollama not available, skipping vector index")
        logger.warning("Semantic search will not be available")

    # Update index state for each processed file
    for chunk_file in chunk_files:
        file_hash = IndexState.compute_file_hash(chunk_file)
        doc_stem = chunk_file.name[:-len("_chunks.json")] if chunk_file.name.endswith("_chunks.json") else chunk_file.stem
        chunk_prefix = f"{doc_stem}_chunk_"
        file_chunks = [c for c in chunks if c.chunk_id.startswith(chunk_prefix)]

        index_types = ["bm25"]
        if vector_indexer.is_available():
            index_types.append("vector")

        index_state.add_document(
            filename=chunk_file.name,
            file_path=str(chunk_file.relative_to(output_path)),
            content_hash=file_hash,
            chunk_count=len(file_chunks),
            index_types=index_types,
        )

    # Update index paths
    index_state.bm25_index_path = "indexes/bm25_index.pkl"
    if vector_indexer.is_available():
        index_state.vector_index_path = "indexes/chroma_db"

    # Save index state
    index_state.save()

    # Save metadata (legacy)
    metadata = {
        "num_chunks": len(chunks),
        "num_documents": len(chunk_files),
        "bm25_build_time": bm25_time,
        "vector_build_time": vector_time if vector_indexer.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = index_dir / "index_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nIndexes saved to {index_dir}")
    logger.info(index_state.summary())
    logger.info("Build complete!")


def show_stats(output_dir: str) -> None:
    """
    Show index statistics.

    Args:
        output_dir: Directory containing indexes
    """
    output_path = Path(output_dir)
    index_dir = output_path / "indexes"
    converted_dir = output_path / "converted"

    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.info(f"Run with --build-index first")
        sys.exit(1)

    # Load metadata
    metadata_path = index_dir / "index_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print("\n=== Index Statistics ===")
        print(f"Total chunks: {metadata['num_chunks']}")
        print(f"Total documents: {metadata['num_documents']}")
        print(f"BM25 build time: {metadata['bm25_build_time']:.2f}s")
        if metadata.get('vector_build_time'):
            print(f"Vector build time: {metadata['vector_build_time']:.2f}s")
        print(f"Last updated: {metadata['timestamp']}")
    else:
        print("Index metadata not found")

    # Show per-document stats
    try:
        _, _, HybridRetriever = _import_indexers()
        retriever = HybridRetriever(
            index_dir=str(index_dir),
            chunks_dir=str(converted_dir)
        )
        stats = retriever.get_stats()

        print(f"\n=== Document Breakdown ===")
        for doc_name, count in sorted(stats["documents"].items()):
            print(f"  {doc_name}: {count} chunks")

        print(f"\n=== Search Capabilities ===")
        print(f"BM25 search: {'✓' if stats['bm25_available'] else '✗'}")
        print(f"Semantic search: {'✓' if stats['semantic_available'] else '✗'}")
        if not stats['semantic_available']:
            print("  (Ollama not running - install and start Ollama for semantic search)")

    except Exception as e:
        logger.error(f"Error loading retriever: {e}")


def search_and_display(
    output_dir: str,
    query: str,
    top_k: int,
    mode: str,
    rerank: bool = False
) -> None:
    """
    Search and display results.

    Args:
        output_dir: Directory containing indexes
        query: Search query
        top_k: Number of results to return
        mode: Search mode (bm25, semantic, hybrid)
        rerank: Whether to apply cross-encoder reranking
    """
    output_path = Path(output_dir)
    index_dir = output_path / "indexes"
    converted_dir = output_path / "converted"

    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.info(f"Run with --build-index first")
        sys.exit(1)

    # Initialize retriever
    try:
        _, _, HybridRetriever = _import_indexers()
        retriever = HybridRetriever(
            index_dir=str(index_dir),
            chunks_dir=str(converted_dir)
        )
    except Exception as e:
        logger.error(f"Error loading retriever: {e}")
        sys.exit(1)

    # Execute search
    start_time = time.time()
    try:
        results = retriever.search(query, top_k=top_k, mode=mode, rerank=rerank)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)

    search_time = time.time() - start_time

    # Display results
    print(f"\n{'=' * 70}")
    print(f"Search Results for \"{query}\"")
    print(f"{'=' * 70}")
    print(f"Mode: {mode}")
    if mode == "hybrid":
        print(f"  (BM25 + semantic vector search)")
    print(f"Found {len(results)} results in {search_time:.2f}s")
    print()

    if not results:
        print("No results found. Try:")
        print("  - Using different keywords")
        print("  - Using broader terms")
        print("  - Checking spelling")
        return

    for result in results:
        separator = '\u2500' * 70
        print(separator)
        print(f"Result {result.rank}/{len(results)} (Score: {result.score:.3f})")
        print(separator)

        # Document info
        chunk = result.chunk
        doc_name = "_".join(chunk.chunk_id.split("_")[:-2])
        print(f"Document: {doc_name}")
        if chunk.source_path:
            print(f"Source: {chunk.source_path}")
        print(f"Type: {chunk.doc_type.value if hasattr(chunk.doc_type, 'value') else chunk.doc_type}")
        print(f"Citation: {chunk.citation_string}")
        print(f"Pages: {', '.join(map(str, chunk.pages))}")
        bates = chunk.citation.get("bates_range", [])
        if bates:
            print(f"Bates: {', '.join(bates)}")

        # Per-line match page attribution
        if chunk.citation.get("page_map"):
            match_page, match_bates = _find_query_page(query, chunk)
            if match_page is not None:
                print(f"Match Page: {match_page}")
                if match_bates:
                    print(f"Match Bates: {match_bates}")

        # Enrichment metadata
        if chunk.category or chunk.relevance_score or chunk.summary:
            enrichment_parts = []
            if chunk.category:
                enrichment_parts.append(f"Category: {chunk.category}")
            if chunk.relevance_score:
                enrichment_parts.append(f"Relevance: {chunk.relevance_score}")
            if chunk.claims_addressed:
                enrichment_parts.append(f"Claims: {chunk.claims_addressed}")
            print(f"Enrichment: {', '.join(enrichment_parts)}")
            if chunk.summary:
                print(f"Summary: {chunk.summary}")

        # Scores
        if result.bm25_score is not None or result.semantic_score is not None or result.reranker_score is not None:
            score_parts = []
            if result.bm25_score is not None:
                score_parts.append(f"BM25: {result.bm25_score:.3f}")
            if result.semantic_score is not None:
                score_parts.append(f"Semantic: {result.semantic_score:.3f}")
            if result.reranker_score is not None:
                score_parts.append(f"Reranker: {result.reranker_score:.3f}")
            print(f"Scores: {', '.join(score_parts)}")

        # Preview
        print(f"\nPreview:")
        preview = chunk.core_text[:400]
        if len(chunk.core_text) > 400:
            preview += "..."
        # Clean up whitespace
        preview = " ".join(preview.split())
        print(f'"{preview}"')
        print()

    print(f"{'=' * 70}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search litigation documents using hybrid BM25 + semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build indexes
  python lit_doc_retriever.py --index-dir tests/pipeline_output --build-index

  # Search with hybrid mode (default)
  python lit_doc_retriever.py --index-dir tests/pipeline_output \\
      --query "TWT technology battery life"

  # Search with BM25 only
  python lit_doc_retriever.py --index-dir tests/pipeline_output \\
      --query "designated corporate representative" --mode bm25

  # Search with cross-encoder reranking
  python lit_doc_retriever.py --index-dir tests/pipeline_output \\
      --query "TWT technology battery life" --rerank

  # Show statistics
  python lit_doc_retriever.py --index-dir tests/pipeline_output --stats
        """
    )

    parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing pipeline output (converted/ and indexes/)"
    )

    parser.add_argument(
        "--query",
        help="Search query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )

    parser.add_argument(
        "--mode",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)"
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking (requires sentence-transformers)"
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build indexes from chunk files"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics"
    )

    parser.add_argument(
        "--config",
        help="Path to retrieval config JSON (for --build-index)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.build_index:
        build_indexes(args.index_dir, args.config)
    elif args.stats:
        show_stats(args.index_dir)
    elif args.query:
        search_and_display(args.index_dir, args.query, args.top_k, args.mode, rerank=args.rerank)
    else:
        parser.print_help()
        print("\nError: Must specify --build-index, --stats, or --query")
        sys.exit(1)


if __name__ == "__main__":
    main()
