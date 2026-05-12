#!/usr/bin/env python3
"""
Run pipeline steps 1-5 on PDF documents:
  1. Convert (Docling) → .md + .json
  2. Post-process → cleaned .md + initial _citations.json
  3. Citation tracking → bbox-based _citations.json (overwrites step 2's)
  4. Chunking → *_chunks.json files
  5. LLM enrichment (optional) → enriched chunks

Usage:
    python run_pipeline.py --input-dir tests/test_docs --output-dir tests/pipeline_output
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

from citation_tracker import CitationTracker
from citation_types import DocumentType
from chunk_documents import chunk_all_documents
from doc_classifier import (
    classify_directory,
    classify_document,
    ProfileStore,
    ClassificationResult,
    is_condensed_transcript,
)
from docling_converter import DoclingConverter
from parallel_processor import process_documents_parallel, get_optimal_worker_count
from pdf_metadata import extract_pdf_metadata, get_page_count
from pipeline_state import PipelineState
from post_processor import PostProcessor
from pymupdf_extractor import is_text_based_pdf, extract_deposition
from text_extractor import is_text_transcript, extract_text_deposition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def should_disable_tqdm():
    """Check if progress bars should be disabled."""
    return (os.environ.get('TQDM_DISABLE', '0') == '1' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            os.environ.get('CI', '').lower() == 'true')


def normalize_stem(name: str) -> str:
    """Normalize a filename stem to lowercase with underscores."""
    import re
    result = name.lower().replace(" ", "_").replace("-", "_")
    result = re.sub(r"_+", "_", result)  # collapse multiple underscores
    return result.strip("_")


# Type sets for handler routing
TRANSCRIPT_TYPES = {DocumentType.DEPOSITION, DocumentType.HEARING_TRANSCRIPT}
PARAGRAPH_TYPES = {
    DocumentType.EXPERT_REPORT, DocumentType.PLEADING,
    DocumentType.DECLARATION, DocumentType.MOTION,
    DocumentType.BRIEF, DocumentType.WITNESS_STATEMENT,
    DocumentType.AGREEMENT,
}


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"


def generate_pipeline_report(
    results: list,
    skipped_condensed: list,
    converted_dir: Path,
    pipeline_elapsed: float,
    cleanup_json: bool = True,
    parallel: bool = False,
    max_workers: int = None,
):
    """Print a concise dashboard-style pipeline report.

    Replaces the old verbose per-document listing with aggregated statistics
    and sections that highlight problems (errors, warnings) while keeping
    successful runs compact.
    """
    lines = []

    def out(text=""):
        lines.append(text)

    # Partition results
    ok_results = [r for r in results if r["status"] == "OK"]
    failed_results = [r for r in results if r["status"] == "FAILED"]
    skipped_results = [r for r in results if r["status"] == "SKIPPED"]

    total = len(ok_results) + len(failed_results) + len(skipped_results)

    # ── Header ────────────────────────────────────────────────────────
    out("=" * 80)
    out(" PIPELINE REPORT")
    out("=" * 80)

    # ── Summary ───────────────────────────────────────────────────────
    out()
    out(" SUMMARY")
    out(" -------")

    condensed_count = len(skipped_condensed)
    out(f" Documents:  {total} total | {len(ok_results)} OK | "
        f"{len(failed_results)} failed | {len(skipped_results)} skipped | "
        f"{condensed_count} condensed")

    # Chunk and citation totals (from OK results only)
    total_chunks = sum(r.get("chunks_count", 0) for r in ok_results)
    total_citations = sum(r.get("citations_count", 0) for r in ok_results)
    if ok_results:
        avg_chunks = total_chunks / len(ok_results)
        avg_citations = total_citations / len(ok_results)
        out(f" Chunks:     {total_chunks:,} across {len(ok_results)} documents "
            f"(avg {avg_chunks:.1f}/doc)")
        out(f" Citations:  {total_citations:,} total (avg {avg_citations:.1f}/doc)")

        coverages = [r.get("coverage_pct", 0.0) for r in ok_results]
        avg_coverage = sum(coverages) / len(coverages)
        out(f" Coverage:   {avg_coverage:.1f}% avg")

    # Timing
    mode_str = "parallel" if parallel else "sequential"
    if parallel and max_workers:
        mode_str += f", {max_workers} workers"
    if ok_results:
        elapsed_vals = [r.get("elapsed_seconds", 0) for r in ok_results if r.get("elapsed_seconds")]
        avg_time = sum(elapsed_vals) / len(elapsed_vals) if elapsed_vals else 0
        out(f" Time:       {_format_duration(pipeline_elapsed)} "
            f"({_format_duration(avg_time)} avg/doc) | {mode_str}")
    else:
        out(f" Time:       {_format_duration(pipeline_elapsed)} | {mode_str}")

    # ── Classification Distribution ───────────────────────────────────
    if ok_results:
        out()
        out(" CLASSIFICATION DISTRIBUTION")
        out(" ---------------------------")

        type_counts: dict[str, int] = {}
        for r in ok_results:
            dt = r.get("doc_type", "unknown")
            type_counts[dt] = type_counts.get(dt, 0) + 1

        for dt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / len(ok_results)
            out(f" {dt:<24s} {count:>4}  ({pct:.1f}%)")

    # ── Extraction Methods ────────────────────────────────────────────
    if ok_results:
        out()
        out(" EXTRACTION METHODS")
        out(" ------------------")

        method_counts: dict[str, int] = {}
        for r in ok_results:
            method = r.get("extraction_method", "docling")
            method_counts[method] = method_counts.get(method, 0) + 1

        method_labels = {"docling": "docling", "pymupdf": "pymupdf (fast path)", "text_extractor": "text_extractor (fast path)"}
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / len(ok_results)
            label = method_labels.get(method, method)
            out(f" {label:<24s} {count:>4}  ({pct:.1f}%)")

    # ── Errors ────────────────────────────────────────────────────────
    if failed_results:
        out()
        out(f" ERRORS ({len(failed_results)} document{'s' if len(failed_results) != 1 else ''})")
        out(" " + "-" * len(f"ERRORS ({len(failed_results)} document{'s' if len(failed_results) != 1 else ''})"))
        for r in failed_results:
            source = r.get("source_path", r["file"])
            elapsed = r.get("elapsed_seconds")
            elapsed_str = f" ({_format_duration(elapsed)})" if elapsed else ""
            out(f" FAIL  {r['file']}{elapsed_str}")
            out(f"       Error: {r.get('error', 'Unknown error')}")
            out(f"       Source: {source}")

    # ── Warnings ──────────────────────────────────────────────────────
    warnings_sections = []

    # Zero citations
    zero_cite = [r for r in ok_results if r.get("citations_count", 0) == 0]
    if zero_cite:
        section_lines = [f" Zero citations ({len(zero_cite)}):"]
        for r in zero_cite:
            section_lines.append(f"   - {r['file']} (type: {r.get('doc_type', 'unknown')})")
        warnings_sections.append(section_lines)

    # Low coverage < 50%
    low_cov = [r for r in ok_results
               if r.get("citations_count", 0) > 0 and r.get("coverage_pct", 0) < 50]
    if low_cov:
        section_lines = [f" Low coverage < 50% ({len(low_cov)}):"]
        for r in low_cov:
            section_lines.append(
                f"   - {r['file']} (type: {r.get('doc_type', 'unknown')}, "
                f"{r['citations_count']} citations, {r['coverage_pct']:.1f}% coverage)")
        warnings_sections.append(section_lines)

    # Zero chunks
    zero_chunks = [r for r in ok_results if r.get("chunks_count", 0) == 0]
    if zero_chunks:
        section_lines = [f" Zero chunks produced ({len(zero_chunks)}):"]
        for r in zero_chunks:
            section_lines.append(
                f"   - {r['file']} (type: {r.get('doc_type', 'unknown')}, "
                f"{r.get('citations_count', 0)} citations)")
        warnings_sections.append(section_lines)

    # Citation degraded (had JSON but citation tracking failed)
    degraded = [r for r in ok_results if r.get("citation_degraded")]
    if degraded:
        section_lines = [f" Citation tracking degraded ({len(degraded)}):"]
        for r in degraded:
            section_lines.append(f"   - {r['file']} (type: {r.get('doc_type', 'unknown')})")
        warnings_sections.append(section_lines)

    # No JSON available
    no_json = [r for r in ok_results if not r.get("had_json", True)]
    if no_json:
        section_lines = [f" No JSON for citation tracking ({len(no_json)}):"]
        for r in no_json:
            section_lines.append(f"   - {r['file']} (type: {r.get('doc_type', 'unknown')})")
        warnings_sections.append(section_lines)

    # Low-confidence classification
    low_conf = [r for r in ok_results
                if r.get("classification_confidence") is not None
                and r["classification_confidence"] < 0.15]
    if low_conf:
        section_lines = [f" Low-confidence classification ({len(low_conf)}):"]
        for r in low_conf:
            section_lines.append(
                f"   - {r['file']} (classified as: {r.get('doc_type', 'unknown')}, "
                f"confidence: {r['classification_confidence']:.2f})")
        warnings_sections.append(section_lines)

    if warnings_sections:
        total_warn = sum(
            len([l for l in s if l.startswith("   - ")]) for s in warnings_sections
        )
        out()
        out(f" WARNINGS ({total_warn} document{'s' if total_warn != 1 else ''})")
        out(" " + "-" * len(f"WARNINGS ({total_warn} document{'s' if total_warn != 1 else ''})"))
        for section_lines in warnings_sections:
            for line in section_lines:
                out(line)
            out()

    # ── Info ──────────────────────────────────────────────────────────
    if skipped_results or skipped_condensed:
        out()
        out(" INFO")
        out(" ----")
        if skipped_results:
            out(f" Skipped: {len(skipped_results)} (already completed from previous run)")
        if skipped_condensed:
            out(f" Condensed transcripts: {len(skipped_condensed)} "
                "(unsupported multi-page layout)")
            for name in skipped_condensed:
                out(f"   - {name}")

    # ── Output ────────────────────────────────────────────────────────
    out()
    out(" OUTPUT")
    out(" ------")
    out(f" Directory: {converted_dir}/")
    try:
        output_files = list(converted_dir.iterdir())
        total_size = sum(f.stat().st_size for f in output_files)

        # Count by extension
        ext_counts: dict[str, int] = {}
        for f in output_files:
            ext = f.suffix
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        ext_summary = " + ".join(
            f"{count} {ext}" for ext, count in sorted(ext_counts.items())
        )
        out(f" Files:     {len(output_files)} ({ext_summary})")

        if total_size >= 1024 * 1024:
            out(f" Size:      {total_size / 1024 / 1024:.1f} MB")
        else:
            out(f" Size:      {total_size / 1024:.1f} KB")
    except FileNotFoundError:
        out(" Files:     (directory not found)")

    if cleanup_json:
        out(" (Docling JSON files cleaned up to save disk space)")

    out("=" * 80)

    # Print the report
    print("\n" + "\n".join(lines))

    # Write failed documents to JSON for easy parsing
    if failed_results:
        failed_json_path = converted_dir / "_failed_documents.json"
        import json as _json
        failed_data = []
        for r in failed_results:
            failed_data.append({
                "file": r["file"],
                "stem": r.get("stem"),
                "source_path": r.get("source_path"),
                "error": r.get("error", "Unknown error"),
                "elapsed_seconds": r.get("elapsed_seconds"),
            })
        try:
            with open(failed_json_path, "w") as f:
                _json.dump(failed_data, f, indent=2)
            print(f"\n Failed document details written to: {failed_json_path}")
        except Exception:
            pass  # Non-critical, don't fail the report


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    use_existing: Path = None,
    cleanup_json: bool = True,
    enrich: bool = False,
    enrich_backend: str = "ollama",
    case_type: str = "patent",
    parties: str = "",
    resume: bool = False,
    force: bool = False,
    skip_failed: bool = True,
    conversion_timeout: int = 300,
    parallel: bool = False,
    max_workers: int = None,
    interactive: bool = True,
):
    """Run steps 1-3 (+ optional enrichment) on all PDFs in input_dir.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for pipeline output
        use_existing: Path to existing converted files (skip Docling)
        cleanup_json: Delete large Docling JSON files after processing (default: True)
        enrich: Whether to run LLM enrichment after chunking
        enrich_backend: LLM backend for enrichment (ollama or anthropic)
        case_type: Case type for enrichment context
        parties: Comma-separated party names for enrichment context
        resume: Resume from previous run using saved state
        force: Force reprocessing even if stage already complete
        skip_failed: Skip documents that have failed multiple times
        conversion_timeout: Timeout in seconds for document conversion (default: 300)
        parallel: Enable parallel processing of documents
        max_workers: Number of parallel workers (default: cpu_count - 1)
    """
    pipeline_start = time.monotonic()
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline state for checkpoint/resume
    state = PipelineState(output_dir)

    if resume:
        logger.info("Resuming from previous run")
        logger.info(state.summary())

    pdfs = sorted(
        p for ext in ("*.pdf", "*.txt", "*.docx")
        for p in input_dir.rglob(ext)
    )
    if not pdfs:
        logger.error("No documents found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d document(s) in %s", len(pdfs), input_dir)

    # ── Pre-scan: Classify all documents ─────────────────────────────
    logger.info("[Pre-scan] Classifying documents with PyMuPDF...")
    profile_store = ProfileStore()
    classifications = classify_directory(
        str(input_dir),
        interactive=interactive,
        profile_store=profile_store,
    )
    # Build doc_type_map: normalized_stem -> DocumentType
    doc_type_map = {stem: cr.doc_type for stem, cr in classifications.items()}

    # ── Filter out condensed transcripts ─────────────────────────────
    skipped_condensed = []
    filtered_pdfs = []
    for pdf_path in pdfs:
        if pdf_path.suffix.lower() == ".pdf" and is_condensed_transcript(str(pdf_path)):
            logger.info("Skipping condensed transcript: %s", pdf_path.name)
            skipped_condensed.append(pdf_path.name)
        else:
            filtered_pdfs.append(pdf_path)

    if skipped_condensed:
        logger.info("Skipped %d condensed transcript(s): %s",
                    len(skipped_condensed), ", ".join(skipped_condensed))
    pdfs = filtered_pdfs

    # Build source_path_map: normalized_stem -> original relative path
    # Detect and disambiguate collisions (e.g., "Foo-Bar.pdf" and "foo_bar.pdf")
    source_path_map = {}
    stem_to_pdf = {}  # normalized_stem -> pdf_path (for collision detection)
    for pdf_path in pdfs:
        normalized = normalize_stem(pdf_path.stem)
        try:
            rel_path = str(pdf_path.relative_to(input_dir))
        except ValueError:
            rel_path = pdf_path.name

        if normalized in source_path_map:
            # Collision: two files map to the same stem
            existing = stem_to_pdf[normalized].name
            suffix = 2
            disambiguated = f"{normalized}_{suffix}"
            while disambiguated in source_path_map:
                suffix += 1
                disambiguated = f"{normalized}_{suffix}"
            logger.warning(
                "Stem collision: '%s' and '%s' both normalize to '%s'. "
                "Renaming second to '%s'.",
                existing, pdf_path.name, normalized, disambiguated,
            )
            normalized = disambiguated

        source_path_map[normalized] = rel_path
        stem_to_pdf[normalized] = pdf_path

    # Build reverse map: pdf_path -> disambiguated stem (used by both parallel and sequential)
    pdf_to_stem = {v: k for k, v in stem_to_pdf.items()}

    # Re-key classifications to disambiguated stems and handle collisions explicitly.
    base_counts = {}
    for pdf_path in pdfs:
        base = normalize_stem(pdf_path.stem)
        base_counts[base] = base_counts.get(base, 0) + 1

    disambiguated_classifications = {}
    for pdf_path in pdfs:
        base = normalize_stem(pdf_path.stem)
        normalized = pdf_to_stem.get(pdf_path, base)

        if pdf_path.suffix.lower() == ".txt":
            cr = ClassificationResult(
                doc_type=DocumentType.DEPOSITION,
                confidence=1.0,
                is_text_based=True,
                needs_user_input=False,
                signals={},
            )
        elif base_counts.get(base, 0) > 1:
            # Prevent stem collisions from collapsing multiple documents into one key.
            cr = classify_document(str(pdf_path), profile_store=profile_store)
        else:
            cr = classifications.get(base)
            if cr is None:
                cr = ClassificationResult(
                    doc_type=DocumentType.UNKNOWN,
                    confidence=0.0,
                    is_text_based=False,
                    needs_user_input=False,
                    signals={},
                )

        disambiguated_classifications[normalized] = cr
        doc_type_map[normalized] = cr.doc_type

    classifications = disambiguated_classifications

    logger.info(
        "Classifications: %s",
        {stem: cr.doc_type.value for stem, cr in classifications.items()},
    )

    # Check if parallel processing is requested
    if parallel:
        if max_workers is None:
            max_workers = get_optimal_worker_count()

        logger.info("Using parallel processing with %d workers", max_workers)

        # Build path-keyed normalized stems map to avoid same-stem collisions.
        normalized_stems = {
            str(pdf.resolve()): pdf_to_stem.get(pdf, normalize_stem(pdf.stem))
            for pdf in pdfs
        }

        # Process documents in parallel
        results = process_documents_parallel(
            pdfs=pdfs,
            output_dir=output_dir,
            normalized_stems=normalized_stems,
            classifications=classifications,
            state=state,
            conversion_timeout=conversion_timeout,
            cleanup_json=cleanup_json,
            use_existing=use_existing,
            max_workers=max_workers,
            force=force,
            skip_failed=skip_failed,
        )

        # Skip to enrichment step
        logger.info("Parallel processing complete")

    else:
        # Sequential processing (original code)
        results = []
        disable_progress = should_disable_tqdm()
        pbar = tqdm(pdfs, desc="Processing documents", unit="doc", disable=disable_progress)

        for pdf_path in pbar:
            stem = pdf_path.stem
            normalized = pdf_to_stem.get(pdf_path, normalize_stem(stem))

            # Get document state (with source info for error reporting)
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

            # Skip if document completed and not forcing
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

            # Skip if document failed too many times
            if skip_failed and doc_state.status == "failed" and doc_state.retry_count >= 3:
                logger.info("Skipping %s (failed %d times: %s)",
                           pdf_path.name, doc_state.retry_count, doc_state.error)
                results.append({
                    "file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "stem": normalized,
                    "status": "SKIPPED",
                    "reason": f"Failed {doc_state.retry_count} times",
                })
                continue

            doc_start = time.monotonic()
            logger.info("=" * 60)
            logger.info("Processing: %s", pdf_path.name)
            pbar.set_postfix_str(f"{pdf_path.name[:30]}...")

            try:
                known_type = doc_type_map.get(normalized)
                cr = classifications.get(normalized)
                is_text = cr.is_text_based if cr else False

                # ── Text transcript fast path ────────────────────────────
                if (pdf_path.suffix.lower() == ".txt"
                    and known_type in TRANSCRIPT_TYPES
                    and is_text_transcript(str(pdf_path))
                    and state.should_process_document(normalized, "conversion", force)):

                    logger.info("[TextExtractor] Plain-text transcript detected")
                    try:
                        txt_result = extract_text_deposition(str(pdf_path), str(converted_dir))
                        logger.info(
                            "  Extracted %d lines, %d citations",
                            txt_result["line_count"],
                            txt_result["citation_count"],
                        )

                        doc_state.mark_stage_complete("conversion")
                        doc_state.mark_stage_complete("post_processing")
                        doc_state.mark_stage_complete("citation_tracking")
                        doc_state.mark_completed()
                        state.save()

                        results.append({
                            "file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "stem": normalized,
                            "status": "OK",
                            "doc_type": (known_type or DocumentType.DEPOSITION).value,
                            "md_file": Path(txt_result["md_path"]).name,
                            "json_file": None,
                            "citations_count": txt_result["citation_count"],
                            "coverage_pct": 100.0,
                            "type_distribution": {"transcript_line": txt_result["citation_count"]},
                            "extraction_method": "text_extractor",
                            "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                            "classification_confidence": cr.confidence if cr else None,
                            "classification_needs_input": cr.needs_user_input if cr else False,
                            "had_json": False,
                            "citation_degraded": False,
                            "chunks_count": 0,
                            "bates_gaps": [],
                            "bates_duplicates": [],
                            "line_gaps": [],
                        })
                        continue

                    except Exception as e:
                        error_msg = f"Text extraction failed: {str(e)}"
                        logger.error(error_msg)
                        doc_state.mark_failed(error_msg)
                        state.save()
                        results.append({
                            "file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "stem": normalized,
                            "status": "FAILED",
                            "error": error_msg,
                            "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                        })
                        continue

                # ── PyMuPDF fast path for text-based depositions ─────────────
                # Require strong classifier confidence; otherwise the fast path
                # silently swallows patent apps and scholarly PDFs whose
                # monospace text triggers a weak deposition signal.
                _conf = cr.confidence if cr else 0.0
                if (known_type in TRANSCRIPT_TYPES and _conf >= 0.30 and
                    is_text and is_text_based_pdf(str(pdf_path)) and
                    state.should_process_document(normalized, "conversion", force)):

                    logger.info("[PyMuPDF] Text-based deposition detected, using direct extraction")
                    try:
                        pymupdf_result = extract_deposition(str(pdf_path), str(converted_dir))
                        logger.info(
                            "  Extracted %d lines, %d citations",
                            pymupdf_result["line_count"],
                            pymupdf_result["citation_count"],
                        )

                        # If PyMuPDF extracted 0 citations, fall back to Docling
                        if pymupdf_result["citation_count"] == 0:
                            logger.warning("PyMuPDF extracted 0 citations, falling back to Docling")
                            raise Exception("No citations extracted - non-standard page markers")

                        # Mark all stages complete for PyMuPDF extraction
                        doc_state.mark_stage_complete("conversion")
                        doc_state.mark_stage_complete("post_processing")
                        doc_state.mark_stage_complete("citation_tracking")
                        doc_state.mark_completed()
                        state.save()

                        results.append({
                            "file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "stem": normalized,
                            "status": "OK",
                            "doc_type": (known_type or DocumentType.DEPOSITION).value,
                            "md_file": Path(pymupdf_result["md_path"]).name,
                            "json_file": None,
                            "citations_count": pymupdf_result["citation_count"],
                            "coverage_pct": 100.0,
                            "type_distribution": {"transcript_line": pymupdf_result["citation_count"]},
                            "extraction_method": "pymupdf",
                            "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                            "classification_confidence": cr.confidence if cr else None,
                            "classification_needs_input": cr.needs_user_input if cr else False,
                            "had_json": False,
                            "citation_degraded": False,
                            "chunks_count": 0,
                            "bates_gaps": [],
                            "bates_duplicates": [],
                            "line_gaps": [],
                        })
                        continue

                    except Exception as e:
                        logger.error("PyMuPDF extraction failed: %s", e)
                        logger.info("Falling back to Docling conversion")
                        # Fall through to Docling conversion

            except Exception as e:
                error_msg = f"Pre-processing error: {str(e)}"
                logger.error(error_msg)
                doc_state.mark_failed(error_msg)
                state.save()
                results.append({
                    "file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                    "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                })
                continue

            # ── Step 1: Conversion ───────────────────────────────────────
            if pdf_path.suffix.lower() == ".txt":
                # Text files should have been handled by the text fast path
                error_msg = f"Text file {pdf_path.name} was not handled by text extractor"
                logger.error(error_msg)
                doc_state.mark_failed(error_msg)
                state.save()
                results.append({
                    "file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                    "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                })
                continue

            if state.should_process_document(normalized, "conversion", force):
                logger.info("[Step 1] Converting with Docling...")

                try:
                    # Check if we should use pre-existing converted files
                    existing_json = None
                    existing_md = None
                    if use_existing:
                        existing_json = use_existing / f"{normalized}.json"
                        existing_md = use_existing / f"{normalized}.md"

                    if use_existing and existing_json.exists() and existing_md.exists():
                        logger.info("  Using existing converted files from %s", use_existing)
                        # Copy to our output dir
                        target_json = converted_dir / f"{normalized}.json"
                        target_md = converted_dir / f"{normalized}.md"
                        if not target_json.exists():
                            shutil.copy2(existing_json, target_json)
                        if not target_md.exists():
                            shutil.copy2(existing_md, target_md)
                        # Also copy bates sidecar if present
                        existing_bates = use_existing / f"{normalized}_bates.json"
                        if existing_bates.exists():
                            target_bates = converted_dir / f"{normalized}_bates.json"
                            if not target_bates.exists():
                                shutil.copy2(existing_bates, target_bates)

                        md_path = str(target_md)
                        conversion_citations = {}
                        conversion_errors = []
                    else:
                        converter = DoclingConverter(timeout=conversion_timeout)
                        result = converter.convert_document(str(pdf_path), str(converted_dir))
                        md_path = result.md_path
                        conversion_citations = result.citations_found
                        conversion_errors = result.errors

                        if conversion_errors:
                            logger.warning("  Conversion errors: %s", conversion_errors)
                            if not md_path:
                                error_msg = conversion_errors[0] if conversion_errors else "No output produced"
                                logger.error("  FAILED: %s", error_msg)
                                doc_state.mark_failed(error_msg)
                                state.save()
                                results.append({
                                    "file": pdf_path.name,
                                    "source_path": str(pdf_path),
                                    "stem": normalized,
                                    "status": "FAILED",
                                    "error": error_msg,
                                    "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                                })
                                continue

                        # Docling may produce files with original stem; rename if needed
                        docling_md = converted_dir / f"{stem}.md"
                        docling_json = converted_dir / f"{stem}.json"
                        target_md = converted_dir / f"{normalized}.md"
                        target_json = converted_dir / f"{normalized}.json"

                        if docling_md.exists() and docling_md != target_md:
                            docling_md.rename(target_md)
                            md_path = str(target_md)
                        if docling_json.exists() and docling_json != target_json:
                            docling_json.rename(target_json)

                    # Verify we have the files we need
                    final_md = converted_dir / f"{normalized}.md"
                    final_json = converted_dir / f"{normalized}.json"

                    if not final_md.exists():
                        error_msg = f"No .md file at {final_md}"
                        logger.error("  %s", error_msg)
                        doc_state.mark_failed(error_msg)
                        state.save()
                        results.append({
                            "file": pdf_path.name,
                            "source_path": str(pdf_path),
                            "stem": normalized,
                            "status": "FAILED",
                            "error": error_msg,
                            "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                        })
                        continue

                    has_json = final_json.exists()
                    logger.info("  Output: %s%s",
                               final_md.name,
                               f" + {final_json.name}" if has_json else " (no JSON)")

                    # Mark conversion stage complete
                    doc_state.mark_stage_complete("conversion")
                    state.save()

                except Exception as e:
                    error_msg = f"Conversion error: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    doc_state.mark_failed(error_msg)
                    state.save()
                    results.append({
                        "file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "stem": normalized,
                        "status": "FAILED",
                        "error": error_msg,
                        "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                    })
                    continue
            else:
                logger.info("[Step 1] Skipping conversion (already complete)")
                final_md = converted_dir / f"{normalized}.md"
                final_json = converted_dir / f"{normalized}.json"
                has_json = final_json.exists()
                conversion_citations = {}

            # ── Step 2: Post-processing ──────────────────────────────────
            if state.should_process_document(normalized, "post_processing", force):
                logger.info("[Step 2] Post-processing markdown...")

                try:
                    doc_type = doc_type_map.get(normalized, DocumentType.UNKNOWN)
                    logger.info("  Document type: %s", doc_type.value)

                    processor = PostProcessor()
                    proc_result = processor.process(str(final_md), doc_type)
                    logger.info("  Post-processor citation coverage: %d elements", proc_result.citation_coverage)

                    # Mark post-processing stage complete
                    doc_state.mark_stage_complete("post_processing")
                    state.save()

                except Exception as e:
                    error_msg = f"Post-processing error: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    doc_state.mark_failed(error_msg)
                    state.save()
                    results.append({
                        "file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "stem": normalized,
                        "status": "FAILED",
                        "error": error_msg,
                        "elapsed_seconds": round(time.monotonic() - doc_start, 1),
                    })
                    continue
            else:
                logger.info("[Step 2] Skipping post-processing (already complete)")
                # Detect doc type even if skipping
                doc_type = doc_type_map.get(normalized, DocumentType.UNKNOWN)

            # ── Step 3: Citation tracking (bbox-based) ───────────────────
            citation_degraded = False
            if has_json and state.should_process_document(normalized, "citation_tracking", force):
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
                    if metrics.line_gaps:
                        logger.warning("  Line gaps: %s", metrics.line_gaps[:3])

                    # Mark citation tracking stage complete
                    doc_state.mark_stage_complete("citation_tracking")
                    state.save()

                    # Cleanup: Delete large Docling JSON file (keep _citations.json)
                    if cleanup_json and final_json.exists():
                        json_size = final_json.stat().st_size
                        final_json.unlink()
                        logger.info("  Cleaned up Docling JSON (%d bytes freed)", json_size)

                except Exception as e:
                    error_msg = f"Citation tracking error: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    # Don't fail the document, just log warning
                    logger.warning("  Continuing without bbox-based citations")
                    citations = {}
                    metrics = None
                    citation_degraded = True
                    # Still mark stage complete (partial success)
                    doc_state.mark_stage_complete("citation_tracking")
                    state.save()

            elif not has_json:
                logger.warning("[Step 3] Skipped — no JSON file for bbox-based reconstruction")
                citations = {}
                metrics = None
                doc_state.mark_stage_complete("citation_tracking")
                state.save()
            else:
                logger.info("[Step 3] Skipping citation tracking (already complete)")
                # Load existing citations if available
                citations_file = converted_dir / f"{normalized}_citations.json"
                if citations_file.exists():
                    with open(citations_file) as f:
                        citations = json.load(f)
                    metrics = None  # Don't recompute metrics
                else:
                    citations = {}
                    metrics = None

            # Mark document as completed (all main stages done)
            doc_state.mark_completed()
            state.save()

            # Collect results
            results.append({
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
                "classification_confidence": cr.confidence if cr else None,
                "classification_needs_input": cr.needs_user_input if cr else False,
                "had_json": has_json,
                "citation_degraded": citation_degraded,
                "chunks_count": 0,
                "bates_gaps": metrics.bates_gaps if metrics else [],
                "bates_duplicates": metrics.bates_duplicates if metrics else [],
                "line_gaps": metrics.line_gaps if metrics else [],
            })

    # ── Step 4: Chunking ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[Step 4] Chunking documents...")

    try:
        all_chunks = chunk_all_documents(str(converted_dir), doc_type_map=doc_type_map, source_path_map=source_path_map)

        # Save chunks to individual files
        for stem, chunks in all_chunks.items():
            chunks_file = converted_dir / f"{stem}_chunks.json"
            chunk_dicts = [chunk.to_dict() for chunk in chunks]

            with open(chunks_file, 'w') as f:
                json.dump(chunk_dicts, f, indent=2)

            logger.info("  Saved %d chunks to %s", len(chunks), chunks_file.name)

        logger.info("  Total: %d documents, %d chunks",
                   len(all_chunks), sum(len(chunks) for chunks in all_chunks.values()))

        # Back-fill chunk counts into results
        chunk_counts = {stem: len(chunks) for stem, chunks in all_chunks.items()}
        for r in results:
            if r["status"] == "OK":
                r["chunks_count"] = chunk_counts.get(r["stem"], 0)

    except Exception as e:
        logger.error("Chunking failed: %s", str(e))
        logger.debug(traceback.format_exc())

    # ── Step 5 (optional): LLM Enrichment ────────────────────────────
    if enrich:
        logger.info("=" * 60)
        logger.info("[Step 5] LLM Enrichment (backend: %s)", enrich_backend)

        try:
            from llm_enrichment import LLMEnricher, CaseContext

            case_context = CaseContext(
                case_type=case_type,
                parties=[p.strip() for p in parties.split(",") if p.strip()] if parties else [],
            )

            enricher = LLMEnricher(backend=enrich_backend, delay_between_calls=0.1)
            if enricher.is_available():
                enrich_stats = enricher.enrich_directory(str(converted_dir), case_context)
                logger.info("Enrichment complete: %s", enrich_stats.summary())

                # Mark enrichment complete for all successfully processed documents
                for result in results:
                    if result["status"] == "OK":
                        doc_state = state.get_document(result["stem"], result["file"])
                        doc_state.mark_stage_complete("enrichment")
                state.save()
            else:
                logger.warning("Enrichment backend '%s' not available, skipping.", enrich_backend)

        except Exception as e:
            logger.error("Enrichment error: %s", e)
            logger.debug(traceback.format_exc())
            logger.warning("Continuing without enrichment")

    # ── Final state save ──────────────────────────────────────────────
    state.save()
    logger.info("\n" + state.summary())

    # ── Pipeline Report ──────────────────────────────────────────────
    pipeline_elapsed = time.monotonic() - pipeline_start
    generate_pipeline_report(
        results=results,
        skipped_condensed=skipped_condensed,
        converted_dir=converted_dir,
        pipeline_elapsed=pipeline_elapsed,
        cleanup_json=cleanup_json,
        parallel=parallel,
        max_workers=max_workers,
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline steps 1-3 on PDF documents",
        epilog="""
Examples:
  # Process all documents
  python run_pipeline.py --input-dir tests/test_docs --output-dir output/

  # Resume from previous run
  python run_pipeline.py --input-dir tests/test_docs --output-dir output/ --resume

  # Force reprocess all documents
  python run_pipeline.py --input-dir tests/test_docs --output-dir output/ --force

  # Increase timeout for large documents
  python run_pipeline.py --input-dir docs/ --output-dir output/ --conversion-timeout 600
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Directory for pipeline output")
    parser.add_argument("--use-existing", default=None,
                        help="Path to existing converted files (skip Docling conversion)")
    parser.add_argument("--cleanup-json", action="store_true", default=True,
                        help="Delete Docling JSON files after processing (default: True)")
    parser.add_argument("--no-cleanup-json", dest="cleanup_json", action="store_false",
                        help="Keep Docling JSON files after processing")
    parser.add_argument("--enrich", action="store_true", default=False,
                        help="Run LLM enrichment on chunks after processing")
    parser.add_argument("--enrich-backend", choices=["ollama", "anthropic"], default="ollama",
                        help="LLM backend for enrichment (default: ollama)")
    parser.add_argument("--case-type", default="patent",
                        help="Case type for enrichment context (default: patent)")
    parser.add_argument("--parties", default="",
                        help="Comma-separated party names for enrichment context")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from previous run (skip already-completed documents)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force reprocessing even if stages already complete")
    parser.add_argument("--no-skip-failed", dest="skip_failed", action="store_false", default=True,
                        help="Retry documents that have failed multiple times")
    parser.add_argument("--conversion-timeout", type=int, default=300,
                        help="Timeout in seconds for document conversion (default: 300)")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Enable parallel processing of documents")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    parser.add_argument("--non-interactive", dest="interactive", action="store_false", default=True,
                        help="Skip interactive prompts for low-confidence classifications")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_existing = Path(args.use_existing) if args.use_existing else None

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    run_pipeline(
        input_dir, output_dir, use_existing,
        cleanup_json=args.cleanup_json,
        enrich=args.enrich,
        enrich_backend=args.enrich_backend,
        case_type=args.case_type,
        parties=args.parties,
        resume=args.resume,
        force=args.force,
        skip_failed=args.skip_failed,
        conversion_timeout=args.conversion_timeout,
        parallel=args.parallel,
        max_workers=args.max_workers,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
