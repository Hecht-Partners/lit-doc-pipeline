# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a litigation document processing pipeline that converts legal documents (PDFs, DOCX, plain-text transcripts) into structured, searchable formats optimized for LLM-assisted legal analysis. The system preserves precise citation information (page numbers, Bates stamps, line numbers, column numbers, paragraph numbers) required for legal work.

**Current Status:** ✅ **FULLY OPERATIONAL** - Phases 1-5 complete + Production hardening + Performance optimization + User experience. Core pipeline functional with high-quality citation tracking (99.2% paragraph, 84.5% column detection), chunking, hybrid search, reranking, and optional LLM enrichment. Benchmark: 98% Precision@5. Performance: 3-4x faster with parallel processing, 30x faster with incremental indexing. Progress bars provide visual feedback during all long-running operations. Plain-text (.txt) court reporter transcripts are now supported as first-class input alongside PDFs.

**Last Tested:** 2026-02-18 - 292 tests (230 passing, 62 skipped). Successfully processed 454-document Kindler Fuja corpus (16,113 chunks) with dashboard-style end-of-run report.

## How to Run the Pipeline (Operator Guide)

When the user asks you to process documents, follow this workflow.

### 1. Questions to Ask Before Running

Before starting, gather these inputs (ask only what's missing — don't re-ask if the user already provided them):

| Question | Why it matters | Default |
|----------|---------------|---------|
| **Input directory** (where are the PDFs/TXTs?) | Required — no default | — |
| **Output directory** | Where processed output goes. For new projects use `~/Dev/processed-lit-docs/<project_name>/`. For existing projects, use the project's existing output folder. | `~/Dev/processed-lit-docs/<project_name>/` |
| **Case type** (`patent`, `contract`, `employment`, etc.) | Affects enrichment context and citation format expectations | `patent` |
| **Party names** (comma-separated) | Used by enrichment to identify relevant entities | `""` |
| **Parallel or sequential?** | Parallel is 3-4x faster but uses more memory | `--parallel --max-workers 4` |
| **Interactive or non-interactive?** | Interactive prompts for low-confidence classifications; non-interactive skips them | Interactive |
| **Conversion timeout** | Large scanned PDFs can take 10+ minutes in Docling | `300` (seconds); use `600-1800` for large corpora |
| **Build search indexes after?** | BM25 + vector indexes are built automatically after processing; use `--no-index` to skip | Auto (default) |
| **Run LLM enrichment?** | Adds summaries/categories/relevance — slow, optional | No |

### 2. Recommended Command (Large Corpus)

For a production run on hundreds of documents:

```bash
.venv/bin/python lit_pipeline.py process \
  <INPUT_DIR> \
  <OUTPUT_DIR> \
  --parallel \
  --max-workers 4 \
  --case-type <CASE_TYPE> \
  --parties "<PARTY1>, <PARTY2>" \
  --cleanup-json \
  --conversion-timeout 600 \
  --non-interactive
```

For small test runs or when the user wants to review classifications:

```bash
.venv/bin/python lit_pipeline.py process \
  <INPUT_DIR> \
  <OUTPUT_DIR> \
  --case-type <CASE_TYPE> \
  --cleanup-json
```

### 3. Indexing (Automatic)

Search indexes (BM25 + vector) are now built automatically after processing. No separate step needed.

To skip auto-indexing, add `--no-index` to the process command. To rebuild indexes independently:

```bash
.venv/bin/python lit_pipeline.py index <OUTPUT_DIR>/ [--force-rebuild]
```

**Known issue:** Ollama embedding can return 500 errors on some chunks (especially large tables or non-text content). The indexer skips those chunks gracefully — BM25 still indexes everything. If Ollama is not running, only BM25 will be built. BM25 alone achieves 98% Precision@5, so this is fine for most use cases.

### 4. What to Show the User at the End

The pipeline now prints a **dashboard-style report** automatically. After the run completes, review the report and highlight to the user:

1. **ERRORS section** — Any documents that failed (timeouts, conversion errors). Tell the user which files failed and why. Suggest increasing `--conversion-timeout` if timeouts dominate.
2. **WARNINGS section** — Quality issues that need attention:
   - **Zero citations** — Documents with no citation data (may be cover letters, image-only pages, or conversion issues)
   - **Low coverage < 50%** — Partial citation extraction (often scanned/degraded PDFs)
   - **Zero chunks** — Documents that produced no searchable content
   - **Low-confidence classification** — Documents the classifier was unsure about (confidence < 0.15)
   - **Citation tracking degraded** — JSON was present but citation extraction threw an error
3. **SUMMARY line** — Total OK/failed/skipped counts, chunk and citation totals, timing
4. **Offer next steps:**
   - "Want me to run a test search to verify results?"
   - "Want me to re-run the N failed documents with a higher timeout?"
   - "Want me to remove any specific documents from the corpus?"

### 5. Search Tips

- **BM25 mode** (`--mode bm25`) is fastest and most reliable — use for keyword/phrase searches
- **Hybrid mode** (`--mode hybrid`) combines BM25 + vector similarity — requires Ollama
- **Reranking** (`--rerank`) improves precision but adds ~19s/query latency
- Search result previews only show the first ~500 characters of a chunk — the match may be deeper in the text. Use `--top-k 10` for broader coverage.

## Critical Requirements

### 1. Citation Accuracy is PARAMOUNT
- Every chunk must be traceable to specific pages, lines, columns, or Bates numbers
- Citations must match legal citation format:
  - Depositions: `"Daniel Alexander Dep. 45:12-18"` (page:line range)
  - Patents: `"'152 Patent, col. 3:45-55"` (column:line range)
  - Expert Reports: `"Fuja Report ¶ 42"` (paragraph number)
- Never discard line numbers before associating them with text content

### 2. No Garbage Text from Images
- ALWAYS use these Docling flags:
  ```bash
  --image-export-mode placeholder
  --no-enrich-picture-classes
  --no-enrich-picture-description
  --no-enrich-chart-extraction
  ```
- NEVER disable OCR (`--no-ocr`) - needed for scanned documents
- NEVER use `--image-export-mode embedded` (creates massive base64 strings)

### 3. Dual-Format Processing Required
- Extract BOTH JSON and Markdown from Docling:
  - JSON: Contains page metadata, bbox coordinates, provenance data
  - Markdown: Contains readable text
- Use `--to md --to json` flag
- JSON is needed for citation reconstruction, can be deleted after with `--cleanup-json`

## Architecture Overview

The pipeline consists of 5 main steps:

1. **Conversion (Docling / TextExtractor / PyMuPDF)** - Extract to JSON + MD with proper flags (PDFs via Docling or PyMuPDF fast path; plain-text transcripts via TextExtractor)
2. **Post-Processing** - Clean OCR artifacts while preserving citation markers
3. **Citation Reconstruction** - Parse line/column/paragraph numbers from JSON provenance or text structure
4. **Chunking** - Create semantic chunks inheriting citation metadata (saves to *_chunks.json)
5. **Search Indexing** - Hybrid BM25 + Chroma with metadata (runs automatically after step 4)
6. **LLM Enrichment (Optional)** - Summaries, key quotes, categorization (optional flag)

**Note:** Steps 1-5 run via `lit_pipeline.py process` (use `--no-index` to skip step 5). Step 6 via `--enrich` flag or `lit_pipeline.py enrich`. Indexes can also be rebuilt independently via `lit_pipeline.py index`.

**Additional commands:** `lit_pipeline.py classify` (standalone classification), `lit_pipeline.py remove` (surgical document removal from indexes).

## Implementation Status

### ✅ Phase 1-5: FULLY OPERATIONAL

**Phase 1: Citation Foundation**
- `citation_tracker.py` - Bbox-based line/column/paragraph inference
- `pymupdf_extractor.py` - Fast path for text-based PDF depositions
- `text_extractor.py` - Fast path for plain-text (.txt) court reporter transcripts
- `format_handlers.py` - Document type detection
- Citation coverage: 100% for text depositions, 99.8% Bates coverage

**Phase 2: Core Pipeline**
- `post_processor.py` - Text cleaning with citation preservation
- `chunk_documents.py` - Section-aware chunking with Q/A preservation
- Footnote inline insertion for all legal documents
- ✅ **FIXED (2026-02-13):** Chunking step now integrated into `run_pipeline.py` (was missing)

**Phase 3: Vector Search**
- `bm25_indexer.py` - TF-IDF keyword search (<10ms queries)
- `vector_indexer.py` - Chroma + Ollama nomic-embed-text
- `hybrid_retriever.py` - RRF score fusion
- `lit_doc_retriever.py` - Legacy CLI (use `lit_pipeline.py` instead)

**Phase 4: Cross-Encoder Reranker**
- `reranker.py` - BAAI/bge-reranker-v2-m3 with lazy loading (8K token context)
- Integrated into hybrid retriever with `--rerank` flag
- Graceful degradation when sentence-transformers not installed

**Phase 5: LLM Enrichment**
- `llm_enrichment.py` - Three backends: Ollama, Anthropic, Claude Code
- Quote validation, category/relevance validation, claims filtering
- CLI integration: `--enrich`, `--enrich-backend`, `--case-type`, `--parties`

**Document Classification & Management**
- `doc_classifier.py` - Generic PyMuPDF-based classifier with 5-signal system (filename, content, structural, font, learned profiles)
- Interactive prompting by default for low-confidence classifications (shows first-page text + file path)
- Self-learning via `ProfileStore` (persists to `~/.lit-pipeline/type_profiles.json`)
- `lit_pipeline.py remove` - Surgical document removal from all indexes without full rebuild

**Fixes Applied (2026-02-13):**
1. ✅ Added missing chunking step (Step 4) to `run_pipeline.py`
2. ✅ Fixed indentation bug causing sequential loop to run after parallel processing
3. ✅ Updated step numbering (enrichment now Step 5, was Step 4)
4. ✅ Added `chunk_all_documents` import and integration

## Production Features

### ✅ Production Readiness (Complete)
- ✅ Unified CLI entry point (`lit_pipeline.py` with 7 subcommands)
- ✅ Config file support (`config_loader.py`, JSON/YAML)
- ✅ Benchmark script (`benchmark.py`, 20 queries, Precision@5 + latency)
- ✅ Error handling and recovery (`pipeline_state.py`, checkpoint/resume)
- ✅ Automatic retry limiting (max 3 attempts per document)
- ✅ Graceful failure handling (continue on errors)

### ✅ Performance Optimization (Complete)
- ✅ Parallel document processing (`parallel_processor.py`, 3-4x faster)
- ✅ Incremental indexing (`index_state.py`, 30x faster when unchanged)
- ✅ Configurable timeouts (`--conversion-timeout`, default 300s)
- ✅ Worker count control (`--parallel --max-workers N`)
- ✅ SHA256-based change detection for incremental builds

### ✅ User Experience (Complete)
- ✅ Progress bars with tqdm (visual feedback for long operations)
- ✅ Sequential processing: document count and ETA
- ✅ Nested progress: file + chunk levels during enrichment
- ✅ Parallel processing: real-time completion updates
- ✅ Auto-disables in test environments (PYTEST_CURRENT_TEST detection)

### Quality Improvements
- ~~Paragraph number extraction~~ → ✅ **99.2% coverage** (supports numbered format "N. ")
- ~~Column detection~~ → ✅ **84.5% on spec pages** (exceeds goal)
- ~~Bates sequential validation~~ → ✅ **Implemented** (gap and duplicate detection)
- Claim-aware chunking for patents (future enhancement)

## Quick Start: Running the Full Pipeline

### Step 1: Process Documents (Steps 1-4: Convert, Post-process, Citations, Chunking)

**Output lives outside this repo** — default location is `~/Dev/processed-lit-docs/<project_name>/`.

```bash
# RECOMMENDED: Parallel processing (3-4x faster)
.venv/bin/python lit_pipeline.py process \
  tests/test_docs \
  ~/Dev/processed-lit-docs/my_case \
  --parallel \
  --max-workers 4 \
  --case-type patent \
  --parties "Intel, Proxim" \
  --cleanup-json \
  --conversion-timeout 600
```

**What this does:**
- Converts PDFs via Docling (JSON + Markdown); plain-text transcripts via TextExtractor fast path
- Extracts citations (line numbers, columns, paragraphs, Bates)
- Post-processes text (cleans OCR, inlines footnotes)
- **Creates semantic chunks** (saves to *_chunks.json files)
- Uses parallel processing (4 workers)
- Cleans up JSON files after processing
- **Builds search indexes automatically** (BM25 + vector via Ollama)

**Expected output:** `~/Dev/processed-lit-docs/my_case/converted/` with .md, _citations.json, and **_chunks.json** files, plus `indexes/bm25_index.pkl` and `indexes/chroma_db/`

### Step 2: Test Search
```bash
# Hybrid search with reranking (best quality)
.venv/bin/python lit_pipeline.py search \
  ~/Dev/processed-lit-docs/my_case/ \
  "TWT technology wireless networking" \
  --mode hybrid \
  --rerank \
  --top-k 5

# BM25-only search (fastest)
.venv/bin/python lit_pipeline.py search \
  ~/Dev/processed-lit-docs/my_case/ \
  "patent claim construction" \
  --mode bm25 \
  --top-k 5

# View statistics
.venv/bin/python lit_pipeline.py stats ~/Dev/processed-lit-docs/my_case/
```

### Step 4: Remove a Document (Optional)

To surgically remove a processed document from all output files and search indexes:

```bash
# Remove by document stem (normalized filename)
.venv/bin/python lit_pipeline.py remove ~/Dev/processed-lit-docs/my_case/ "daniel_alexander_10_24_2025"

# Also accepts original filenames (auto-normalized to stem)
.venv/bin/python lit_pipeline.py remove ~/Dev/processed-lit-docs/my_case/ "Daniel Alexander - 10-24-2025"
```

**What this does:**
- Deletes per-document files (.md, _citations.json, _chunks.json) from `<output>/converted/`
- Removes matching entries from the BM25 index (filters sparse matrix rows)
- Removes matching entries from the ChromaDB vector index (`collection.delete()`)
- Cleans up pipeline state and index state JSON files
- No full index rebuild required — changes are surgical

### Alternative Commands

```bash
# Sequential processing (slower, less memory)
.venv/bin/python lit_pipeline.py process tests/test_docs ~/Dev/processed-lit-docs/my_case/ --case-type patent

# Non-interactive processing (skip prompts for low-confidence classifications)
.venv/bin/python lit_pipeline.py process tests/test_docs ~/Dev/processed-lit-docs/my_case/ --non-interactive

# Resume after interruption
.venv/bin/python lit_pipeline.py process tests/test_docs ~/Dev/processed-lit-docs/my_case/ --resume

# Force reprocessing (ignore state)
.venv/bin/python lit_pipeline.py process tests/test_docs ~/Dev/processed-lit-docs/my_case/ --force

# Force rebuild indexes
.venv/bin/python lit_pipeline.py index ~/Dev/processed-lit-docs/my_case/ --force-rebuild

# Classify documents without processing
.venv/bin/python lit_pipeline.py classify tests/test_docs/
.venv/bin/python lit_pipeline.py classify tests/test_docs/ --non-interactive
.venv/bin/python lit_pipeline.py classify tests/test_docs/ --show-profiles

# Optional: LLM enrichment (adds summaries, categories, relevance scores)
.venv/bin/python lit_pipeline.py enrich ~/Dev/processed-lit-docs/my_case/converted/ --backend ollama
```

## Key Data Structures

### Context Card Schema
```json
{
  "card_id": "unique-id",
  "source": "document.pdf",
  "doc_type": "deposition|patent|expert_report|pleading|exhibit",
  "core_text": "...",
  "pages": [14, 15],
  "citation": {
    "pdf_pages": [14, 15],
    "transcript_lines": {"14": [5, 25], "15": [1, 12]},
    "column_lines": {"column": 3, "lines": [45, 55]},
    "paragraph_numbers": [42],
    "bates_range": ["INTEL_PROX_00001784"]
  },
  "citation_string": "Daniel Alexander Dep. 14:5-15:12",
  "category": "witness_statement",
  "relevance_score": "high|medium|low"
}
```

### Citation Metadata File (`_citations.json`)
```json
{
  "#/texts/42": {
    "page": 14,
    "line_start": 5,
    "line_end": 12,
    "bates": "INTEL_PROX_00001784",
    "type": "transcript_line"
  }
}
```

## Common Pitfalls to Avoid

1. **NEVER strip line numbers before extracting them** - Extract citations before any text cleaning
2. **NEVER split Q&A pairs when chunking depositions** - Implement deposition-aware chunking
3. **NEVER assume key quotes from LLM are verbatim** - Always validate with exact substring match
4. **NEVER confuse patent numbers (7+ digits) with claim numbers (1-2 digits)**
5. **NEVER disable JSON output** - Page metadata is required for citation tracking
6. **ALWAYS run chunking after citation extraction** - Chunks need citation metadata to be useful
7. **Indexing runs automatically** - Use `--no-index` only if you have a reason to skip it

## Document Type Handling

### Depositions
- Detect: Q&A format, line numbers 1-25 per page
- Accepts both PDF and plain-text (.txt) court reporter transcripts
- Plain-text transcripts use `text_extractor.py` fast path (no Docling needed)
- Citations: page:line format (e.g., "45:12-18")
- Chunking: Never split Q&A pairs, preserve line ranges

### Patents
- Detect: Column layout, "col." markers
- Citations: column:line format (e.g., "col. 3:45-55")
- Handle both physical columns (bbox) and inline citations

### Expert Reports
- Detect: ¶ or § markers, "Paragraph N" text
- Citations: paragraph format (e.g., "¶ 42")
- Track paragraph ranges for multi-paragraph sections

### Pleadings
- Detect: Caption, numbered paragraphs, signature blocks
- Citations: paragraph format (e.g., "Complaint ¶ 23")
- Extract caption and party information

## Technical Stack

### Core Dependencies
```
docling>=2.70.0              # PDF conversion
pymupdf>=1.24.0              # PDF extraction
scikit-learn>=1.5.0          # BM25/TF-IDF
chromadb>=0.4.0              # Vector store
numpy>=2.0.0                 # Numerical ops
tqdm>=4.67.0                 # Progress bars
```

### Optional Dependencies (Installed)
```
sentence-transformers>=3.0.0  # Cross-encoder reranking (installed: 5.2.2, model: bge-reranker-v2-m3)
anthropic>=0.40.0             # Claude API for enrichment (installed: 0.46.0)
```

### External Services (Optional)
```bash
# Ollama for embeddings and enrichment
ollama pull nomic-embed-text    # 274MB
ollama pull llama3.1:8b          # 4.7GB
```

## File Structure

```
lit-doc-pipeline/
├── lit_pipeline.py                 # ⭐ MAIN CLI - Use this! (7 subcommands)
├── benchmark.py                    # Search quality benchmark (Precision@K)
├── lit_pipeline/                   # Package: all internal modules
│   ├── __init__.py
│   ├── run_pipeline.py             # Pipeline orchestration (steps 1-4)
│   ├── parallel_processor.py       # Parallel document processing
│   ├── pipeline_state.py           # Error handling & checkpoint/resume
│   ├── index_state.py              # Incremental indexing state
│   ├── config_loader.py            # JSON/YAML config support
│   ├── docling_converter.py        # PDF/DOCX conversion via Docling
│   ├── citation_tracker.py         # Citation reconstruction from Docling JSON
│   ├── citation_types.py           # Data structures (Chunk, SearchResult, etc.)
│   ├── pymupdf_extractor.py        # Fast path for text-based PDF depositions
│   ├── text_extractor.py           # Fast path for plain-text (.txt) transcripts
│   ├── doc_classifier.py           # Generic document type classifier (self-learning)
│   ├── format_handlers.py          # Document type detection
│   ├── post_processor.py           # Text cleaning + footnote insertion
│   ├── chunk_documents.py          # Section-aware chunking (chunk_all_documents)
│   ├── pdf_metadata.py             # PDF metadata helpers
│   ├── bm25_indexer.py             # TF-IDF keyword search
│   ├── vector_indexer.py           # ChromaDB + Ollama embeddings
│   ├── hybrid_retriever.py         # RRF score fusion
│   ├── reranker.py                 # Cross-encoder reranking (bge-reranker-v2-m3, 8K context)
│   ├── lit_doc_retriever.py        # Legacy CLI / index-builder (use lit_pipeline.py instead)
│   └── llm_enrichment.py           # LLM enrichment (3 backends)
├── configs/
│   ├── default_config.json         # Chunking, Bates patterns, Docling settings
│   ├── retrieval_config.json       # Search configuration
│   └── enrichment_config.json      # LLM backend settings
├── tests/
│   ├── test_docs/                  # 7 test PDFs (87MB)
│   ├── test_citation_tracker.py
│   ├── test_chunk_documents.py
│   ├── test_hybrid_retriever.py
│   ├── test_text_extractor.py
│   └── ... (292 tests total)
└── _Archive/
    └── LITIGATION_DOCUMENT_PIPELINE_TRD.md  # Complete spec
```

**Note on `lit_pipeline.py` (script) vs `lit_pipeline/` (package):** Python
disambiguates: running `python lit_pipeline.py` executes the script as
`__main__`, while `from lit_pipeline.foo import X` imports from the
package directory. The script never imports its own filename.

**Key Entry Points:**
- `lit_pipeline.py` - Main CLI at repo root (use this for all operations)
- `lit_pipeline.run_pipeline` - Pipeline orchestration imported by the CLI
- `lit_pipeline.chunk_documents` - Contains `chunk_all_documents()` function

## Testing Strategy

### Unit Tests Required
- Citation reconstruction for each document type
- Chunking preserves citation metadata
- Post-processor doesn't destroy citations
- BM25 scoring accuracy
- Score fusion correctness

### Integration Tests Required
- End-to-end pipeline on sample documents
- Search relevance on known queries
- Parallel enrichment
- Citation string generation

### Regression Tests Required
- No garbage text from images
- Page numbers preserved
- Bates numbers extracted correctly
- Line numbers associated with text

## Quality Metrics

- **Citation Accuracy:** 100% of chunks must have valid citation strings
- **Search Relevance:** Top-5 results should include target document 95%+ of time
- **Processing Speed:** < 5 minutes per 100 pages
- **False Positive Rate:** < 10% for keyword categorization
- **Storage Efficiency:** < 500 MB per 1,000 pages (with cleanup-json)

### Benchmark Results (as of 2026-02-11)
- **BM25 Precision@5:** 98.0% (20 queries, 56-chunk corpus)
- **BM25+Rerank Precision@5:** 98.0% (cross-encoder BAAI/bge-reranker-v2-m3)
- **Hit Rate:** 100% (both modes)
- **BM25 Latency:** <1ms per query
- **Rerank Latency:** ~19s per query (bge-reranker-v2-m3 with 30 candidates on MPS)
- Run `benchmark.py` to reproduce

## Litigation Skills Suite

Nine skills in `~/.claude/skills/` consume pipeline output to produce litigation work product with traceable citations. Each skill runs Phase 0 (pipeline discovery) to locate chunk files and search indexes, then uses the search CLI for evidence retrieval.

| Slash Command | Skill File | Purpose |
|---------------|-----------|---------|
| `/draft-motion` | `draft-motion.md` | Motions, oppositions, replies (full intent-locking protocol) |
| `/memo-to-file` | `memo-to-file.md` | Internal memos (strategy, research, case eval, claim analysis) |
| `/depo-prep` | `depo-prep.md` | Deposition outlines, questions, exhibits, impeachment |
| `/draft-pleading` | `draft-pleading.md` | Complaints, answers, counterclaims, ITC complaints |
| `/discovery-draft` | `discovery-draft.md` | Interrogatories, RFPs, RFAs (offensive/defensive) |
| `/discovery-respond` | `discovery-respond.md` | Discovery responses and objections (offensive/defensive) |
| `/review-production` | `doc-review.md` | Production document review charts and memoranda |
| `/build-context` | `reusable-context-creator.md` | Portable context file (≤150k tokens) from knowledge base |
| `/build-manifest` | `case-manifest.md` | Comprehensive case manifest from founding documents |

Skills default to patent litigation conventions unless `case_context.json` or user input indicates otherwise. All factual citations use the chunk's `citation_string` verbatim.

## Reference Documents

- Complete specification: `_Archive/LITIGATION_DOCUMENT_PIPELINE_TRD.md`
- See Section 4 for citation tracking algorithms
- See Section 9.4 for LLM enrichment validation
- See Section 11 for known issues and critical fixes
- See Appendix A for sample document formats
