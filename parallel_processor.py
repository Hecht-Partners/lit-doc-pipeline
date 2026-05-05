"""
Parallel document processing for the litigation pipeline.

Enables concurrent processing of multiple documents while maintaining
thread-safe state tracking and error handling.
"""

import logging
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from citation_tracker import CitationTracker
from citation_types import DocumentType
from doc_classifier import ClassificationResult
from docling_converter import DoclingConverter
from pdf_metadata import extract_pdf_metadata, get_page_count
from pipeline_state import PipelineState
from post_processor import PostProcessor
from pymupdf_extractor import is_text_based_pdf, extract_deposition
from text_extractor import is_text_transcript, extract_text_deposition

# Type sets for handler routing
TRANSCRIPT_TYPES = {DocumentType.DEPOSITION, DocumentType.HEARING_TRANSCRIPT}

logger = logging.getLogger(__name__)


def _should_disable_tqdm():
    """Check if progress bars should be disabled."""
    return (os.environ.get('TQDM_DISABLE', '0') == '1' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            os.environ.get('CI', '').lower() == 'true')


def _path_key(path: Path) -> str:
    """Build a stable key for path-based mappings across processes."""
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def process_single_document(
    pdf_path: Path,
    output_dir: Path,
    normalized: str,
    doc_type: DocumentType,
    is_text_based: bool,
    conversion_timeout: int,
    cleanup_json: bool,
    use_existing: Optional[Path] = None,
    classification_confidence: float = None,
    classification_needs_input: bool = False,
) -> Dict:
    """
    Process a single document through the pipeline.

    This function is designed to be called in a separate process.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        normalized: Normalized document stem
        doc_type: Pre-computed document type from classifier
        is_text_based: Whether the document is text-based (from classifier)
        conversion_timeout: Timeout for conversion
        cleanup_json: Whether to cleanup JSON files
        use_existing: Path to existing converted files
        classification_confidence: Confidence score from classifier (0.0-1.0)
        classification_needs_input: Whether classification needed user input

    Returns:
        Result dictionary with processing status
    """
    doc_start = time.monotonic()
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    try:
        logger.info("Processing: %s (worker PID: %d)", pdf_path.name, os.getpid())

        # ── Text transcript fast path ────────────────────────────
        if (pdf_path.suffix.lower() == ".txt"
            and doc_type in TRANSCRIPT_TYPES
            and is_text_transcript(str(pdf_path))):

            logger.info("[TextExtractor] Plain-text transcript detected")
            txt_result = extract_text_deposition(str(pdf_path), str(converted_dir))
            logger.info("  Extracted %d lines, %d citations",
                       txt_result["line_count"],
                       txt_result["citation_count"])

            return {
                "file": pdf_path.name,
                "source_path": str(pdf_path),
                "stem": normalized,
                "status": "OK",
                "doc_type": (doc_type or DocumentType.DEPOSITION).value,
                "md_file": Path(txt_result["md_path"]).name,
                "json_file": None,
                "citations_count": txt_result["citation_count"],
                "coverage_pct": 100.0,
                "type_distribution": {"transcript_line": txt_result["citation_count"]},
                "extraction_method": "text_extractor",
                "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                "classification_confidence": classification_confidence,
                "classification_needs_input": classification_needs_input,
                "had_json": False,
                "citation_degraded": False,
                "chunks_count": 0,
                "bates_gaps": [],
                "bates_duplicates": [],
                "line_gaps": [],
            }

        # ── PyMuPDF fast path for text-based depositions ─────────────
        # Require strong classifier confidence; otherwise the fast path
        # silently swallows patent apps and scholarly PDFs whose monospace
        # text triggers a weak deposition signal.
        if (doc_type in TRANSCRIPT_TYPES and classification_confidence >= 0.30
                and is_text_based and is_text_based_pdf(str(pdf_path))):
            logger.info("[PyMuPDF] Text-based deposition detected")
            try:
                pymupdf_result = extract_deposition(str(pdf_path), str(converted_dir))
                logger.info("  Extracted %d lines, %d citations",
                           pymupdf_result["line_count"],
                           pymupdf_result["citation_count"])

                # Match sequential behavior: fall back if extraction produced no usable citations.
                if pymupdf_result["citation_count"] == 0:
                    raise RuntimeError("No citations extracted - non-standard page markers")

                return {
                    "file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "stem": normalized,
                    "status": "OK",
                    "doc_type": (doc_type or DocumentType.DEPOSITION).value,
                    "md_file": Path(pymupdf_result["md_path"]).name,
                    "json_file": None,
                    "citations_count": pymupdf_result["citation_count"],
                    "coverage_pct": 100.0,
                    "type_distribution": {"transcript_line": pymupdf_result["citation_count"]},
                    "extraction_method": "pymupdf",
                    "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                    "classification_confidence": classification_confidence,
                    "classification_needs_input": classification_needs_input,
                    "had_json": False,
                    "citation_degraded": False,
                    "chunks_count": 0,
                    "bates_gaps": [],
                    "bates_duplicates": [],
                    "line_gaps": [],
                }
            except Exception as e:
                logger.warning("PyMuPDF extraction failed: %s", e)
                logger.info("Falling back to Docling conversion")
                # Fall through to Docling path below.

        # ── Step 1: Conversion ───────────────────────────────────────
        logger.info("[Step 1] Converting with Docling...")

        # Check for existing files
        existing_json = None
        existing_md = None
        if use_existing:
            existing_json = use_existing / f"{normalized}.json"
            existing_md = use_existing / f"{normalized}.md"

        if use_existing and existing_json and existing_json.exists() and existing_md.exists():
            logger.info("  Using existing converted files")
            final_md = converted_dir / f"{normalized}.md"
            final_json = converted_dir / f"{normalized}.json"
            if not final_md.exists():
                shutil.copy2(existing_md, final_md)
            if not final_json.exists():
                shutil.copy2(existing_json, final_json)

            existing_bates = use_existing / f"{normalized}_bates.json"
            if existing_bates.exists():
                target_bates = converted_dir / f"{normalized}_bates.json"
                if not target_bates.exists():
                    shutil.copy2(existing_bates, target_bates)
        else:
            converter = DoclingConverter(timeout=conversion_timeout)
            result = converter.convert_document(str(pdf_path), str(converted_dir))

            if result.errors and not result.md_path:
                error_msg = result.errors[0] if result.errors else "Conversion failed"
                logger.error("  FAILED: %s", error_msg)
                return {
                    "file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                    "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                }

            # Rename if needed
            docling_md = converted_dir / f"{stem}.md"
            docling_json = converted_dir / f"{stem}.json"
            target_md = converted_dir / f"{normalized}.md"
            target_json = converted_dir / f"{normalized}.json"

            if docling_md.exists() and docling_md != target_md:
                docling_md.rename(target_md)
            if docling_json.exists() and docling_json != target_json:
                docling_json.rename(target_json)

            final_md = target_md
            final_json = target_json

        # Verify output
        if not final_md.exists():
            error_msg = f"No .md file at {final_md}"
            logger.error("  %s", error_msg)
            return {
                "file": pdf_path.name,
                "source_path": str(pdf_path),
                "stem": normalized,
                "status": "FAILED",
                "error": error_msg,
                "elapsed_seconds": round(time.monotonic() - doc_start, 1),
            }

        has_json = final_json.exists()
        logger.info("  Output: %s%s", final_md.name,
                   f" + {final_json.name}" if has_json else " (no JSON)")

        # ── Step 2: Post-processing ──────────────────────────────────
        logger.info("[Step 2] Post-processing markdown...")
        logger.info("  Document type: %s", doc_type.value)

        processor = PostProcessor()
        proc_result = processor.process(str(final_md), doc_type)
        logger.info("  Post-processor citation coverage: %d elements",
                   proc_result.citation_coverage)

        # ── Step 3: Citation tracking (bbox-based) ───────────────────
        citations = {}
        metrics = None

        if has_json:
            logger.info("[Step 3] Reconstructing citations from JSON bbox data...")
            try:
                tracker = CitationTracker(
                    converted_dir=str(converted_dir),
                    doc_type=doc_type,
                )
                citations = tracker.reconstruct_citations(normalized)
                metrics = tracker.validate(citations)

                logger.info("  Citations: %d items, %.1f%% coverage",
                           metrics.total_items, metrics.coverage_pct)
                logger.info("  Types: %s", metrics.type_distribution)

                # Cleanup JSON
                if cleanup_json and final_json.exists():
                    json_size = final_json.stat().st_size
                    final_json.unlink()
                    logger.info("  Cleaned up Docling JSON (%d bytes freed)", json_size)

            except Exception as e:
                logger.error("Citation tracking error: %s", e)
                logger.warning("  Continuing without bbox-based citations")
        else:
            logger.warning("[Step 3] Skipped — no JSON file")

        # Return result
        citation_degraded = (has_json and metrics is None)
        return {
            "file": pdf_path.name,
            "source_path": str(pdf_path),
            "stem": normalized,
            "status": "OK",
            "doc_type": doc_type.value,
            "md_file": final_md.name,
            "json_file": final_json.name if has_json and final_json.exists() else None,
            "citations_count": len(citations),
            "coverage_pct": metrics.coverage_pct if metrics else 0.0,
            "type_distribution": metrics.type_distribution if metrics else {},
            "extraction_method": "docling",
            "elapsed_seconds": round(time.monotonic() - doc_start, 1),
            "classification_confidence": classification_confidence,
            "classification_needs_input": classification_needs_input,
            "had_json": has_json,
            "citation_degraded": citation_degraded,
            "chunks_count": 0,
            "bates_gaps": metrics.bates_gaps if metrics else [],
            "bates_duplicates": metrics.bates_duplicates if metrics else [],
            "line_gaps": metrics.line_gaps if metrics else [],
        }

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "file": pdf_path.name,
            "source_path": str(pdf_path),
            "stem": normalized,
            "status": "FAILED",
            "error": error_msg,
            "elapsed_seconds": round(time.monotonic() - doc_start, 1),
        }


def process_documents_parallel(
    pdfs: List[Path],
    output_dir: Path,
    normalized_stems: Dict[str, str],
    classifications: Dict[str, "ClassificationResult"],
    state: PipelineState,
    conversion_timeout: int = 300,
    cleanup_json: bool = True,
    use_existing: Optional[Path] = None,
    max_workers: Optional[int] = None,
    force: bool = False,
    skip_failed: bool = True,
) -> List[Dict]:
    """
    Process multiple documents in parallel.

    Args:
        pdfs: List of PDF paths to process
        output_dir: Output directory
        normalized_stems: Mapping of path keys to normalized stems
        classifications: Pre-computed classifications from doc_classifier
        state: Pipeline state for tracking progress
        conversion_timeout: Timeout for conversion
        cleanup_json: Whether to cleanup JSON files
        use_existing: Path to existing converted files
        max_workers: Number of parallel workers (default: cpu_count - 1)
        force: Force reprocessing
        skip_failed: Skip documents that have failed multiple times

    Returns:
        List of result dictionaries
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    logger.info("Processing %d documents with %d workers", len(pdfs), max_workers)

    results = []

    # Filter documents based on state
    pdfs_to_process = []
    for pdf_path in pdfs:
        normalized = normalized_stems.get(_path_key(pdf_path), pdf_path.stem.lower())
        _file_size = pdf_path.stat().st_size if pdf_path.exists() else None

        # Compute content hash for deduplication (skip for .txt files)
        _content_hash = None
        _metadata = {}
        _page_count = None
        if pdf_path.suffix.lower() == ".pdf":
            try:
                _content_hash = state.compute_content_hash(pdf_path)

                # Check for duplicate
                duplicate = state.find_duplicate(_content_hash)
                if duplicate and not force:
                    dup_stem, dup_doc = duplicate
                    logger.warning("⚠️  Duplicate detected: %s is identical to %s",
                                 pdf_path.name, dup_doc.filename)
                    logger.warning("   Skipping duplicate (use --force to process anyway)")
                    results.append({
                        "file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "stem": normalized,
                        "status": "SKIPPED",
                        "reason": f"Duplicate of {dup_doc.filename}",
                        "duplicate_of": dup_stem,
                    })
                    continue

                # Extract PDF metadata
                _metadata = extract_pdf_metadata(pdf_path)
                _page_count = get_page_count(pdf_path)

            except Exception as e:
                logger.warning("Failed to compute hash/metadata for %s: %s", pdf_path.name, e)

        doc_state = state.get_document(
            normalized, pdf_path.name,
            source_path=str(pdf_path),
            file_size_bytes=_file_size,
            content_hash=_content_hash,
            page_count=_page_count,
            author=_metadata.get("author"),
            creation_date=_metadata.get("creation_date"),
            modified_date=_metadata.get("modified_date"),
        )

        # Skip if completed and not forcing
        if not force and doc_state.status == "completed":
            logger.info("Skipping %s (already completed)", pdf_path.name)
            results.append({
                "file": pdf_path.name,
                "source_path": str(pdf_path),
                "stem": normalized,
                "status": "SKIPPED",
                "reason": "Already completed",
            })
            continue

        # Skip if failed too many times
        if skip_failed and doc_state.status == "failed" and doc_state.retry_count >= 3:
            logger.info("Skipping %s (failed %d times)", pdf_path.name, doc_state.retry_count)
            results.append({
                "file": pdf_path.name,
                "source_path": str(pdf_path),
                "stem": normalized,
                "status": "SKIPPED",
                "reason": f"Failed {doc_state.retry_count} times",
            })
            continue

        pdfs_to_process.append(pdf_path)

    if not pdfs_to_process:
        logger.info("No documents to process")
        return results

    logger.info("Processing %d documents (skipped %d)",
               len(pdfs_to_process), len(pdfs) - len(pdfs_to_process))

    # Process documents in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_pdf = {}
        for pdf_path in pdfs_to_process:
            normalized = normalized_stems.get(_path_key(pdf_path), pdf_path.stem.lower())

            cr = classifications.get(normalized)
            dt = cr.doc_type if cr else DocumentType.UNKNOWN
            is_text = cr.is_text_based if cr else False

            future = executor.submit(
                process_single_document,
                pdf_path,
                output_dir,
                normalized,
                dt,
                is_text,
                conversion_timeout,
                cleanup_json,
                use_existing,
                classification_confidence=cr.confidence if cr else None,
                classification_needs_input=cr.needs_user_input if cr else False,
            )
            future_to_pdf[future] = (pdf_path, normalized)

        # Collect results as they complete
        disable_progress = _should_disable_tqdm()
        pbar = tqdm(total=len(future_to_pdf), desc="Processing (parallel)",
                    unit="doc", disable=disable_progress)

        for future in as_completed(future_to_pdf):
            pdf_path, normalized = future_to_pdf[future]

            try:
                result = future.result()
                results.append(result)

                # Update state (thread-safe)
                doc_state = state.get_document(normalized, pdf_path.name)

                if result["status"] == "OK":
                    doc_state.mark_stage_complete("conversion")
                    doc_state.mark_stage_complete("post_processing")
                    doc_state.mark_stage_complete("citation_tracking")
                    doc_state.mark_completed()
                    tqdm.write(f"✓ Completed: {pdf_path.name}")
                    pbar.set_postfix_str(f"✓ {pdf_path.name[:30]}")
                else:
                    doc_state.mark_failed(result.get("error", "Unknown error"))
                    tqdm.write(f"✗ Failed: {pdf_path.name} - {result.get('error')}")
                    pbar.set_postfix_str(f"✗ {pdf_path.name[:30]}")

                # Save state after each document
                state.save()
                pbar.update(1)

            except Exception as e:
                error_msg = f"Worker error: {str(e)}"
                logger.error(error_msg, exc_info=True)

                doc_state = state.get_document(normalized, pdf_path.name)
                doc_state.mark_failed(error_msg)
                state.save()

                results.append({
                    "file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                })
                pbar.update(1)

        pbar.close()

    return results


def get_optimal_worker_count() -> int:
    """
    Get optimal number of workers based on system resources.

    Returns:
        Recommended worker count
    """
    cpu_count = mp.cpu_count()

    # Leave one core for system
    optimal = max(1, cpu_count - 1)

    # Cap at 8 workers to avoid overwhelming the system
    return min(optimal, 8)
