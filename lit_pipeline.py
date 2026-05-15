#!/usr/bin/env python3
"""
Unified CLI for the litigation document processing pipeline.

Usage:
    lit-pipeline process <input-dir> <output-dir> [options]
    lit-pipeline index <output-dir> [options]
    lit-pipeline search <index-dir> <query> [options]
    lit-pipeline stats <index-dir>
    lit-pipeline enrich <chunks-dir> [options]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_process(args):
    """Run the document processing pipeline (Steps 1-3)."""
    from lit_pipeline.run_pipeline import run_pipeline

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_existing = Path(args.use_existing) if args.use_existing else None

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        use_existing=use_existing,
        cleanup_json=args.cleanup_json,
        enrich=args.enrich,
        enrich_backend=args.enrich_backend,
        case_type=args.case_type,
        parties=args.parties,
        resume=getattr(args, 'resume', False),
        force=getattr(args, 'force', False),
        skip_failed=getattr(args, 'skip_failed', True),
        conversion_timeout=getattr(args, 'conversion_timeout', 300),
        parallel=getattr(args, 'parallel', False),
        max_workers=getattr(args, 'max_workers', None),
        interactive=getattr(args, 'interactive', True),
    )

    # Auto-index after processing (default behavior)
    if getattr(args, 'index', True):
        logger.info("Building search indexes (BM25 + vector)...")
        from lit_pipeline.lit_doc_retriever import build_indexes
        build_indexes(str(output_dir), config_path=None, force=False)


def cmd_index(args):
    """Build search indexes from processed chunks."""
    from lit_pipeline.lit_doc_retriever import build_indexes

    output_dir = args.output_dir
    config_path = args.config
    force = getattr(args, 'force_rebuild', False)

    build_indexes(output_dir, config_path, force=force)


def cmd_search(args):
    """Search indexed documents."""
    from lit_pipeline.lit_doc_retriever import search_and_display

    index_dir = args.index_dir
    query = args.query
    top_k = args.top_k
    mode = args.mode
    rerank = args.rerank

    search_and_display(index_dir, query, top_k, mode, rerank)


def cmd_stats(args):
    """Show index statistics."""
    from lit_pipeline.lit_doc_retriever import show_stats

    show_stats(args.index_dir)


def cmd_classify(args):
    """Classify documents by type without processing."""
    from lit_pipeline.doc_classifier import classify_directory, ProfileStore

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    profile_store = ProfileStore()

    if args.show_profiles:
        if not profile_store.has_profiles():
            print("No learned profiles found.")
        else:
            print("\nLearned Document Type Profiles:")
            print("=" * 50)
            for doc_type, prof in profile_store.profiles.items():
                print(f"\n  {doc_type} ({prof['count']} examples)")
                avg = prof.get("avg_fingerprint", {})
                for k, v in sorted(avg.items()):
                    if isinstance(v, float):
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: {v}")
        return

    classifications = classify_directory(
        str(input_dir),
        interactive=args.interactive,
        profile_store=profile_store,
    )

    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    for stem, result in sorted(classifications.items()):
        confidence_marker = "!" if result.needs_user_input else " "
        print(f"  {confidence_marker} {stem:50s} -> {result.doc_type.value:25s} "
              f"(confidence: {result.confidence:.2f}, text: {result.is_text_based})")
    print(f"{'='*60}")
    print(f"  Total: {len(classifications)} documents")
    needs_input = sum(1 for r in classifications.values() if r.needs_user_input)
    if needs_input:
        print(f"  Low confidence (!): {needs_input} (use --non-interactive to skip prompts)")


def cmd_remove(args):
    """Remove a processed document and its index entries."""
    import pickle
    import numpy as np

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error("Output directory not found: %s", output_dir)
        sys.exit(1)

    converted_dir = output_dir / "converted"
    indexes_dir = output_dir / "indexes"

    # Normalize the stem
    from lit_pipeline.run_pipeline import normalize_stem
    stem = normalize_stem(args.stem)

    # ── 1. Find and delete per-document files ────────────────────────
    doc_files = [
        converted_dir / f"{stem}.md",
        converted_dir / f"{stem}_citations.json",
        converted_dir / f"{stem}_chunks.json",
    ]

    # Read chunk IDs from chunks file before deleting (needed for index removal)
    chunks_file = converted_dir / f"{stem}_chunks.json"
    chunk_ids = []
    if chunks_file.exists():
        try:
            with open(chunks_file) as f:
                chunks_data = json.load(f)
            chunk_ids = [c["chunk_id"] for c in chunks_data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not read chunk IDs from %s: %s", chunks_file, e)

    if not chunk_ids:
        # Fall back: any chunk ID starting with the stem
        logger.info("No chunk file found, will match by stem prefix in indexes")

    deleted_files = []
    for f in doc_files:
        if f.exists():
            f.unlink()
            deleted_files.append(f.name)

    if deleted_files:
        print(f"Deleted files: {', '.join(deleted_files)}")
    else:
        print(f"No document files found for stem '{stem}' in {converted_dir}")

    # ── 2. Remove from BM25 index ────────────────────────────────────
    bm25_path = indexes_dir / "bm25_index.pkl"
    bm25_removed = 0
    if bm25_path.exists():
        try:
            with open(bm25_path, "rb") as f:
                index_data = pickle.load(f)

            old_ids = index_data["chunk_ids"]
            old_matrix = index_data["tfidf_matrix"]

            # Find indices to keep
            if chunk_ids:
                remove_set = set(chunk_ids)
                keep_mask = [cid not in remove_set for cid in old_ids]
            else:
                keep_mask = [not cid.startswith(f"{stem}_chunk_") for cid in old_ids]

            bm25_removed = sum(1 for k in keep_mask if not k)

            if bm25_removed > 0:
                keep_indices = [i for i, k in enumerate(keep_mask) if k]
                index_data["chunk_ids"] = [old_ids[i] for i in keep_indices]
                index_data["tfidf_matrix"] = old_matrix[keep_indices]

                with open(bm25_path, "wb") as f:
                    pickle.dump(index_data, f)

                print(f"BM25: removed {bm25_removed} chunks ({len(index_data['chunk_ids'])} remaining)")
            else:
                print("BM25: no matching chunks found")
        except Exception as e:
            logger.error("Failed to update BM25 index: %s", e)
    else:
        print("BM25: no index file found")

    # ── 3. Remove from ChromaDB ──────────────────────────────────────
    chroma_dir = indexes_dir / "chroma_db"
    chroma_removed = 0
    if chroma_dir.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection(name="lit_docs")

            # Get IDs to remove
            if chunk_ids:
                ids_to_remove = chunk_ids
            else:
                # Query all IDs and filter by prefix
                all_data = collection.get()
                ids_to_remove = [cid for cid in all_data["ids"] if cid.startswith(f"{stem}_chunk_")]

            if ids_to_remove:
                # ChromaDB delete has a batch limit, process in batches
                batch_size = 5000
                for i in range(0, len(ids_to_remove), batch_size):
                    batch = ids_to_remove[i:i + batch_size]
                    collection.delete(ids=batch)
                chroma_removed = len(ids_to_remove)
                print(f"ChromaDB: removed {chroma_removed} chunks ({collection.count()} remaining)")
            else:
                print("ChromaDB: no matching chunks found")
        except ImportError:
            print("ChromaDB: not installed, skipping")
        except Exception as e:
            logger.error("Failed to update ChromaDB: %s", e)
    else:
        print("ChromaDB: no index directory found")

    # ── 4. Remove from pipeline state ────────────────────────────────
    from lit_pipeline.pipeline_state import PipelineState
    pipeline_state = PipelineState(output_dir)
    if stem in pipeline_state.documents:
        del pipeline_state.documents[stem]
        pipeline_state.save()
        print(f"Pipeline state: removed '{stem}'")

    # ── 5. Remove from index state ───────────────────────────────────
    from lit_pipeline.index_state import IndexState
    index_state = IndexState(indexes_dir)
    chunks_filename = f"{stem}_chunks.json"
    if chunks_filename in index_state.documents:
        index_state.remove_document(chunks_filename)
        index_state.save()
        print(f"Index state: removed '{chunks_filename}'")

    # ── Summary ──────────────────────────────────────────────────────
    total_removed = len(deleted_files) + bm25_removed + chroma_removed
    if total_removed > 0:
        print(f"\nDone. Removed '{stem}' from pipeline ({len(deleted_files)} files, "
              f"{bm25_removed} BM25 entries, {chroma_removed} vector entries).")
    else:
        print(f"\nNo traces of '{stem}' found in {output_dir}.")


def cmd_enrich(args):
    """Run LLM enrichment on chunk files."""
    from lit_pipeline.llm_enrichment import LLMEnricher, CaseContext

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        logger.error("Chunks directory not found: %s", chunks_dir)
        sys.exit(1)

    case_context = CaseContext(
        case_type=args.case_type,
        parties=[p.strip() for p in args.parties.split(",") if p.strip()] if args.parties else [],
    )

    enricher = LLMEnricher(backend=args.backend, delay_between_calls=args.delay)

    if not enricher.is_available():
        logger.error("Enrichment backend '%s' not available", args.backend)
        sys.exit(1)

    stats = enricher.enrich_directory(str(chunks_dir), case_context, force=args.force)
    print(f"\n{stats.summary()}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lit-pipeline",
        description="Litigation document processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process documents (converts, chunks, and builds search indexes)
  lit-pipeline process tests/test_docs output/

  # Process with enrichment
  lit-pipeline process tests/test_docs output/ --enrich --case-type patent

  # Process without auto-indexing
  lit-pipeline process tests/test_docs output/ --no-index

  # Use config file
  lit-pipeline --config my-config.json process tests/test_docs output/

  # Rebuild indexes separately
  lit-pipeline index output/

  # Search
  lit-pipeline search output/ "TWT technology" --rerank

  # Show stats
  lit-pipeline stats output/

  # Enrich existing chunks
  lit-pipeline enrich output/converted/ --backend ollama
        """
    )

    # Global options (before subcommand)
    parser.add_argument(
        "--config",
        help="Path to config file (JSON or YAML)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Process command ──────────────────────────────────────────────────
    process_parser = subparsers.add_parser(
        "process",
        help="Process PDF documents and build search indexes",
        description="Convert PDFs to markdown, extract citations, create chunks, and build BM25/vector search indexes"
    )
    process_parser.add_argument(
        "input_dir",
        help="Directory containing PDF files"
    )
    process_parser.add_argument(
        "output_dir",
        help="Directory for pipeline output"
    )
    process_parser.add_argument(
        "--use-existing",
        help="Path to existing converted files (skip Docling conversion)"
    )
    process_parser.add_argument(
        "--cleanup-json",
        action="store_true",
        default=True,
        help="Delete Docling JSON files after processing (default: True)"
    )
    process_parser.add_argument(
        "--no-cleanup-json",
        dest="cleanup_json",
        action="store_false",
        help="Keep Docling JSON files"
    )
    process_parser.add_argument(
        "--enrich",
        action="store_true",
        help="Run LLM enrichment after processing"
    )
    process_parser.add_argument(
        "--enrich-backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM backend for enrichment (default: ollama)"
    )
    process_parser.add_argument(
        "--case-type",
        default="patent",
        help="Case type for enrichment context (default: patent)"
    )
    process_parser.add_argument(
        "--parties",
        default="",
        help="Comma-separated party names for enrichment"
    )
    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already-completed documents)"
    )
    process_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if stages already complete"
    )
    process_parser.add_argument(
        "--no-skip-failed",
        dest="skip_failed",
        action="store_false",
        default=True,
        help="Retry documents that have failed multiple times"
    )
    process_parser.add_argument(
        "--conversion-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for document conversion (default: 300)"
    )
    process_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing of documents"
    )
    process_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 1)"
    )
    process_parser.add_argument(
        "--non-interactive",
        dest="interactive",
        action="store_false",
        default=True,
        help="Skip interactive prompts for low-confidence document classifications"
    )
    process_parser.add_argument(
        "--no-index",
        dest="index",
        action="store_false",
        default=True,
        help="Skip building search indexes after processing (by default, BM25 and vector indexes are built automatically)"
    )

    # ── Classify command ────────────────────────────────────────────────
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify documents by type (without processing)",
        description="Classify PDF documents using PyMuPDF-based signals"
    )
    classify_parser.add_argument(
        "input_dir",
        help="Directory containing PDF files"
    )
    classify_parser.add_argument(
        "--non-interactive",
        dest="interactive",
        action="store_false",
        default=True,
        help="Skip interactive prompts for low-confidence classifications"
    )
    classify_parser.add_argument(
        "--show-profiles",
        action="store_true",
        help="Show learned type profiles"
    )

    # ── Index command ────────────────────────────────────────────────────
    index_parser = subparsers.add_parser(
        "index",
        help="Build search indexes from processed chunks",
        description="Build BM25 and vector indexes for fast search"
    )
    index_parser.add_argument(
        "output_dir",
        help="Directory containing pipeline output (with converted/ subdirectory)"
    )
    index_parser.add_argument(
        "--config",
        help="Path to retrieval config JSON"
    )
    index_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild all indexes (ignore incremental state)"
    )

    # ── Search command ───────────────────────────────────────────────────
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed documents",
        description="Search using BM25, semantic, or hybrid mode with optional reranking"
    )
    search_parser.add_argument(
        "index_dir",
        help="Directory containing indexes"
    )
    search_parser.add_argument(
        "query",
        help="Search query"
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    search_parser.add_argument(
        "--mode",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)"
    )
    search_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking"
    )

    # ── Stats command ────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show index statistics",
        description="Display index statistics and document breakdown"
    )
    stats_parser.add_argument(
        "index_dir",
        help="Directory containing indexes"
    )

    # ── Remove command ────────────────────────────────────────────────────
    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove a processed document and its index entries",
        description="Delete document files and surgically remove from BM25/ChromaDB indexes"
    )
    remove_parser.add_argument(
        "output_dir",
        help="Directory containing pipeline output"
    )
    remove_parser.add_argument(
        "stem",
        help="Document stem or filename to remove (e.g., 'daniel_alexander_10_24_2025' or 'Daniel Alexander - 10-24-2025')"
    )

    # ── Enrich command ───────────────────────────────────────────────────
    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Run LLM enrichment on chunks",
        description="Add summaries, categories, and relevance scores to existing chunks"
    )
    enrich_parser.add_argument(
        "chunks_dir",
        help="Directory containing *_chunks.json files"
    )
    enrich_parser.add_argument(
        "--backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM backend (default: ollama)"
    )
    enrich_parser.add_argument(
        "--case-type",
        default="patent",
        help="Case type for context (default: patent)"
    )
    enrich_parser.add_argument(
        "--parties",
        default="",
        help="Comma-separated party names"
    )
    enrich_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enrich already-enriched chunks"
    )
    enrich_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )

    # Parse args
    args = parser.parse_args()

    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config if specified
    if hasattr(args, 'config') and args.config:
        from lit_pipeline.config_loader import ConfigLoader
        config = ConfigLoader(args.config)
        # Make config available to command handlers
        args._config = config
    else:
        args._config = None

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command handlers
    if args.command == "process":
        cmd_process(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
