# Litigation Document Processing Pipeline

A Python-based system for converting legal documents (PDFs, DOCX, etc.) into structured, searchable formats optimized for LLM-assisted legal analysis. The pipeline preserves precise citation information (page numbers, Bates stamps, line numbers) required for legal work.

## Features

- **Accurate Citation Tracking**: Preserves page:line, column:line, and paragraph citations
- **Multi-Format Support**: PDF, DOCX, XLSX, EML, MSG, PPTX
- **Hybrid Search**: BM25 + semantic vector search with optional cross-encoder reranking
- **Configurable Pipeline**: JSON/YAML config files for all pipeline parameters
- **LLM Enrichment**: Optional summaries and categorization
- **Robust Error Handling**: Checkpoint/resume, automatic retry limiting, graceful failure handling
- **High Performance**: Parallel processing (3-4x faster) + incremental BM25 indexing (30x faster) + incremental vector indexing (40-5,000x faster for typical updates)
- **Progress Bars**: Visual feedback with ETAs for all long-running operations (auto-disables in tests)

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install Docling (requires Python 3.10+)
pip install docling

# Optional: Install Ollama for embeddings
# https://ollama.ai/
ollama pull nomic-embed-text    # 274MB
```

## Quick Start

```bash
# Process documents (sequential)
lit-pipeline process tests/test_docs output/

# Process in parallel (3-4x faster on large batches)
lit-pipeline process tests/test_docs output/ --parallel

# Resume after interruption
lit-pipeline process tests/test_docs output/ --resume

# Process with enrichment
lit-pipeline process tests/test_docs output/ --enrich --case-type patent

# Process large documents with extended timeout
lit-pipeline process large_docs/ output/ --conversion-timeout 600

# Build search index (incremental - only reindexes changed files)
lit-pipeline index output/

# Force rebuild all indexes
lit-pipeline index output/ --force-rebuild

# Search with reranking
lit-pipeline search output/ "witness testimony about TWT technology" --rerank

# Show statistics
lit-pipeline stats output/
```

**Note:** Use `.venv/bin/python lit_pipeline.py` if not installed globally.

## `--use-existing` with Parallel Mode

Use `--use-existing <dir>` when Docling conversion has already been done and you want to rerun downstream stages quickly.

Example:

```bash
lit-pipeline process input_docs/ output/ --parallel --use-existing prior_output/converted
```

Behavior:

- Expects normalized filenames in `<dir>`: `{stem}.md` and `{stem}.json` (optional `{stem}_bates.json`).
- In parallel mode, each worker now copies those files into `output/converted` before post-processing/citation/chunking.
- If PyMuPDF fast-path extraction produces zero citations, the worker falls back to Docling for parity with sequential mode.
- Transcript metadata preserves `deposition` vs `hearing_transcript` labels.

## Project Status

**Current Status:** Phases 1-5 Complete ✅

- ✅ **Phase 1-2:** Citation tracking, chunking, post-processing
- ✅ **Phase 3:** Hybrid search (BM25 + semantic vector)
- ✅ **Phase 4:** Cross-encoder reranker
- ✅ **Phase 5:** LLM enrichment (Ollama/Anthropic/Claude Code backends)

**Test Coverage:** 153 tests (137 passing, 16 skipped)

**Citation Quality:**
- Depositions: 100% line-level accuracy
- Expert Reports: 99.2% paragraph detection
- Patents: 84.5% column detection on spec pages
- Bates stamps: Sequential validation with gap detection

**Search Quality (Benchmark):**
- BM25 Precision@5: 98.0% (20 queries, 56-chunk corpus)
- BM25+Rerank Precision@5: 98.0% (cross-encoder `ms-marco-MiniLM-L-6-v2`)
- Hit Rate: 100% (at least one relevant result in every top-5)
- BM25 Latency: <1ms | Rerank Latency: ~443ms

## Benchmarking

Run the search quality benchmark against the test corpus:

```bash
# Default: 20 queries, Precision@5, BM25 vs BM25+Rerank
.venv/bin/python benchmark.py

# Custom top-K
.venv/bin/python benchmark.py --top-k 10

# Custom chunks directory
.venv/bin/python benchmark.py --chunks-dir path/to/chunks

# Save detailed JSON results
.venv/bin/python benchmark.py --output results.json
```

The benchmark measures Precision@K, hit rate, and per-query latency across BM25-only and BM25+Rerank modes. Results are saved to `benchmark_results.json`.

**Documentation:**
- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture and design
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Error handling & recovery guide
- [PERFORMANCE.md](PERFORMANCE.md) - Performance optimization guide
- [NEXT_STEPS.md](NEXT_STEPS.md) - Roadmap and remaining work
- [CLAUDE.md](../CLAUDE.md) - Implementation guidance

## License

MIT
