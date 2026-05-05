# Changelog

All notable changes to the litigation document pipeline.

---

## [1.1.1] - 2026-03-07

### Fixed: Parallel Processing + `--use-existing`

- `--parallel` now uses path-keyed stem mapping to avoid same-stem collisions across nested folders.
- Worker processes now copy `--use-existing` artifacts (`{stem}.md`, `{stem}.json`, optional `{stem}_bates.json`) into `output/converted`.
- Parallel PyMuPDF fast-path now matches sequential behavior: if extraction yields zero citations, it falls back to Docling.
- Transcript fast-path metadata now preserves `hearing_transcript` vs `deposition` instead of forcing deposition.
- Fixed stale `page_count` metadata leakage between documents in both sequential and parallel loops.
- Classification results are re-keyed to disambiguated stems used by the pipeline run, improving consistency in chunking and reporting.

**Files:** `run_pipeline.py`, `parallel_processor.py`, `tests/test_parallel_processor.py`

---

## [1.1.0] - 2026-03-04

### Critical Bug Fixes - Citation Engine

Four systematic bugs in `chunk_documents.py` that caused citation inaccuracies have been fixed:

#### Fixed: Line Numbers Off by 1-2 Lines
- **Bug:** Line ranges tracked against wrong page variable
- **Impact:** Deposition citations like `175:17-19` were wrong (should be `175:15-19`)
- **Fix:** Changed `current_page` → `page` in line_ranges tracking (lines 256-260)

#### Fixed: Page Numbers Completely Wrong
- **Bug:** Chunk overlap lost metadata, causing stale page references
- **Impact:** Citations referenced wrong pages (e.g., `148:19-21` instead of `141:10-16`)
- **Fix:** Preserve full metadata for overlap lines (lines 289-313)

#### Fixed: Paragraph Content Mismatches
- **Bug:** Limited paragraph detection + no hard token limit
- **Impact:** Paragraph numbers cited wrong section content
- **Fix:** Enhanced regex patterns (¶, §, Paragraph N) + max_tokens safety (lines 395-420)

#### Enhanced: Bates Number Citation Format
- **Previous:** `Document, pp. 2-3 [BATES_001]`
- **New:** `Document at BATES_001-BATES_002` (legal convention)
- **Fix:** Citations prefer Bates numbers over page numbers (lines 710-723)

**Testing:** Re-chunked 868 documents, verified citation accuracy on test cases

---

### Added: Deduplication

- SHA256 content hashing prevents duplicate document processing
- Automatic skip with warning for identical files
- Force override available with `--force` flag
- **Files:** `pipeline_state.py`, `pdf_metadata.py`, `run_pipeline.py`, `parallel_processor.py`

---

### Added: PDF Metadata Extraction

- Extracts author, creation date, modified date from PDF metadata
- Stored in pipeline state for all documents
- Enables timeline analysis and author attribution
- **Files:** `pdf_metadata.py`, `pipeline_state.py`

---

### Added: Incremental Vector Indexing

- Only re-embeds chunks from modified documents (vs full rebuild)
- **Performance:** 40-5,000x speedup for typical updates
- Auto-detection with fallback to full rebuild on errors
- **Files:** `vector_indexer.py`, `lit_doc_retriever.py`

**Benchmarks:**
- 1 changed doc (41 chunks): 16 hours → 10 seconds (5,760x faster)
- 132 changed docs (4,192 chunks): 16 hours → 25 minutes (38x faster)

---

## [1.0.0] - 2026-02-11

### Initial Release

- Document conversion with Docling
- Citation tracking (page, line, Bates, paragraph)
- Type-specific chunking (13 document types)
- Hybrid BM25 + vector search
- Cross-encoder reranking
- Parallel processing
- Incremental BM25 indexing
- LLM enrichment (optional)

---

## Migration Guide

### Upgrading to v1.1.0

**Citation Fixes:**
- Existing projects need to re-chunk documents to apply fixes:
  ```bash
  # Back up current chunks
  cp -r output/converted output/converted_backup

  # Delete old chunks
  rm output/converted/*_chunks.json

  # Re-chunk with fixed code
  lit-pipeline index output/ --force-rebuild
  ```

**Deduplication:**
- Automatically enabled, no configuration required
- Existing projects will dedupe on next processing run

**Metadata Extraction:**
- Automatically enabled, no configuration required
- Existing projects: metadata will be extracted on next processing run

---

## Compatibility

- **Python:** 3.10-3.13 (ChromaDB has issues with 3.14)
- **Ollama:** Required for vector search (optional)
- **PyMuPDF:** Required for PDF processing
- **Docling:** Required for conversion
