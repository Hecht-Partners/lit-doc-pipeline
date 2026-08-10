# Changelog

All notable changes to the litigation document pipeline.

---

## [1.3.0] - 2026-08-10

### Added: Load-File Ingest as the Preferred Path for Vendor Productions

Productions delivered with Concordance load files (.dat/.opt) and extracted
TEXT no longer go through Docling/OCR at all — vendor text is higher
fidelity and orders of magnitude faster. New `lit_pipeline.py loadfile`
subcommand ingests one production or a whole delivery root; `process`
auto-detects .dat load files and routes to it (opt out with
`--no-loadfile`).

- **Page-boundary recovery (`page_align.py`)** so chunks carry per-page
  Bates citations even though vendor TEXT is document-level. Per-document
  routing: vendor form feeds when they match the produced page count
  (authoritative); otherwise fast per-page Tesseract OCR of the production
  TIFs is used as an alignment fingerprint — globally-unique token/bigram
  anchors, longest-non-decreasing-subsequence monotonicity (a garbled page
  cannot cascade), and bounded local refinement of wide gaps. Benchmarked
  against vendor ground truth (form feeds stripped and realigned, 700
  scanned tax-document pages): weighted mean page Jaccard 0.982, 95.9% of
  pages ≥ 0.90.
- **Citation-safety contract:** documents that cannot be paginated safely
  (form-feed count contradicts the produced page count — e.g. text extracted
  from a 4-up W2 native — or no images/tesseract available, or low alignment
  confidence) are ingested with document-level Bates cites and listed in
  `flagged_pagination.json`; a page is never guessed.
- **Header aliases** in `loadfile_ingest.parse_dat`: accepts both
  PRODBEG/PRODEND/PAGES/TEXTPATH and BEGDOC/ENDDOC/PGCOUNT/TEXTFILE
  vocabularies (extensible table).
- **Opticon .opt parsing** (`page_align.parse_opt`) for per-page image
  paths and document breaks; load-file path resolution tolerates delivery
  layouts where paths are relative to the root, the production dir, or a
  doubled production dir (`REPROD001/REPROD001/`).
- **Multi-production ingest** (`ingest_tree`/`find_productions`): discovers
  every .dat under a delivery root (including a `load files/` dir),
  de-duplicated by production stem.
- Per-page OCR is cached (`<corpus>/.ocr_cache` by default, `--ocr-cache`
  to override); pagination can be skipped entirely with `--no-paginate`.

**Files:** `lit_pipeline/page_align.py` (new), `lit_pipeline/loadfile_ingest.py`,
`lit_pipeline.py`, `tests/test_page_align.py` (new),
`prototypes/loadfile-page-align/` (validation prototype + benchmark)

---

## [1.2.0] - 2026-08-03

### Added: Precise Per-Page Pagination in Chunks

Chunks previously exposed pages only as a chunk-level min-max range in
`citation_string` (e.g. `pp. 1-10` for a chunk actually spanning pages 1, 6,
8-10), forcing downstream reviewers to infer a quote's page from text
position — typically off by at least one page. Pagination is now explicit
and per-line:

- **Inline `[PAGE:N]` markers in `core_text`** at every page transition
  (`[PAGE:N | BATES]` when the page's Bates stamp is known), so any quote's
  page is directly readable. Skipped for transcripts, which already cite
  transcript page:line.
- **`citation.page_spans`** — compact `{page, line_start, line_end}` runs
  (1-indexed into `core_text`) computed from the per-line `page_map`;
  `page_map`/`bates_map` remain aligned 1:1 with `core_text` lines after
  marker insertion.
- **Precise `citation_string` page lists** — gaps preserved
  (`pp. 1, 6, 8-10`), never a min-max span. Generic Bates citations sort
  stamps (`Doc at ABBOTT0000055-ABBOTT0000060`).
- **Per-page Bates derivation from filename ranges**
  (`DocumentChunker.derive_bates_by_page`): productions named
  `{BEGBATES}-{ENDBATES} title.pdf` get one stamp per page when no Bates was
  extracted from footers and the range exactly matches the page count.
- **Separator-less Bates footer pattern** (`ABBOTT0000123`-style) added to
  `citation_tracker` and `docling_converter`.

**Files:** `chunk_documents.py`, `citation_tracker.py`,
`docling_converter.py`, `tests/test_page_precision.py`,
`tests/test_page_attribution.py`

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
