#!/usr/bin/env python3
"""
Benchmark script for search quality and reranker performance.

Measures:
- Precision@K for known queries across BM25-only and BM25+rerank
- Per-query latency and rerank overhead
- Result quality comparison between search modes

Usage:
    .venv/bin/python benchmark.py                          # Run against default test corpus
    .venv/bin/python benchmark.py --chunks-dir path/to/chunks --index-dir path/to/index
    .venv/bin/python benchmark.py --rebuild-index           # Force index rebuild
    .venv/bin/python benchmark.py --top-k 10                # Precision@10 instead of @5
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from lit_pipeline.bm25_indexer import BM25Indexer
from lit_pipeline.citation_types import Chunk, DocumentType, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark query definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    """A query with known relevant documents/chunks for evaluation."""
    query: str
    description: str
    # Expected document prefixes that should appear in top-K results.
    # A result matches if its chunk_id starts with any of these prefixes.
    expected_doc_prefixes: List[str]
    # Optional: specific chunk IDs that are the best answer
    ideal_chunk_ids: List[str] = field(default_factory=list)


# Queries designed against the test corpus (Alexander deposition + Cole report).
# TWT is the central technology in this patent case, so both documents discuss
# it extensively. Queries are tagged with ALL documents that contain relevant
# content to avoid penalizing the reranker for surfacing cross-document results.
BENCHMARK_QUERIES = [
    # --- Deposition-focused queries (Alexander) ---
    BenchmarkQuery(
        query="BIOS configuration enable TWT",
        description="Alexander testified there is no way to enable TWT through BIOS",
        expected_doc_prefixes=["daniel_alexander"],
    ),
    BenchmarkQuery(
        query="Intel source code target wait time drivers",
        description="Alexander discussed familiarity with Intel source code for TWT",
        expected_doc_prefixes=["daniel_alexander"],
    ),
    BenchmarkQuery(
        query="Chrome OS TWT accidentally enabled Linux",
        description="Discussion of accidental TWT enablement on Chrome OS and Linux",
        expected_doc_prefixes=["daniel_alexander"],
    ),
    BenchmarkQuery(
        query="Harrison peak standalone chip module CRF",
        description="Alexander described Harrison Peak hardware architecture",
        expected_doc_prefixes=["daniel_alexander"],
    ),
    BenchmarkQuery(
        query="deposition designated corporate representative testimony",
        description="Deposition procedural language unique to Alexander transcript",
        expected_doc_prefixes=["daniel_alexander"],
    ),
    BenchmarkQuery(
        query="Glenn Cox Guy DeMari Chrome OS",
        description="Specific Intel employees Alexander discussed",
        expected_doc_prefixes=["daniel_alexander"],
    ),
    # --- Expert report-focused queries (Cole) ---
    BenchmarkQuery(
        query="damages royalty rate calculation methodology",
        description="Cole report damages analysis methodology",
        expected_doc_prefixes=["2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="Dr. Eric Cole qualifications cybersecurity expert Lockheed Martin",
        description="Cole's background and qualifications",
        expected_doc_prefixes=["2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="incremental value asserted claims 152 Patent apportionment",
        description="Cole report on value of asserted patent claims",
        expected_doc_prefixes=["2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="Cisco identified six key features WiFi 6 OFDMA MU-MIMO",
        description="Cole report cites Cisco's identified key features of WiFi 6",
        expected_doc_prefixes=["2025_12_11_cole_report"],
    ),
    # --- Cross-document queries (TWT is discussed in both) ---
    BenchmarkQuery(
        query="target wake time TWT technology power saving",
        description="Core technology at issue — discussed in both documents",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="802.11 AX WiFi 6 standard certification",
        description="WiFi 6 / 802.11ax standard — discussed in both documents",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="broadcast TWT feature WiFi",
        description="TWT feature broadly — both documents address it",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="register key test mode TWT enabled disabled",
        description="TWT enable/disable — both documents reference this",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="restricted TWT WiFi 7 new feature improvements",
        description="WiFi 7 and TWT improvements — mentioned in both documents",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="WiFi Alliance certification program testing",
        description="WiFi Alliance — referenced by both deponent and expert",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="battery life power saving improvement percentage",
        description="Battery life benefits of TWT — both documents",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="Proxense LLC Intel Corporation patent infringement",
        description="Case parties — present in both documents",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="Intel WiFi products wireless capability OEM",
        description="Intel WiFi products — discussed in both",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
    BenchmarkQuery(
        query="power management WiFi devices accused products",
        description="Power management — core to both documents",
        expected_doc_prefixes=["daniel_alexander", "2025_12_11_cole_report"],
    ),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result of running a single benchmark query."""
    query: str
    description: str
    mode: str
    top_k: int
    precision: float
    latency_ms: float
    result_chunk_ids: List[str]
    result_scores: List[float]
    hit: bool  # At least one expected doc found in top-K


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    total_chunks: int
    num_queries: int
    top_k: int
    modes: Dict[str, Dict] = field(default_factory=dict)
    query_results: List[QueryResult] = field(default_factory=list)

    def add_mode_summary(self, mode: str, results: List[QueryResult]):
        precisions = [r.precision for r in results]
        latencies = [r.latency_ms for r in results]
        hits = [r.hit for r in results]
        self.modes[mode] = {
            "mean_precision": sum(precisions) / len(precisions) if precisions else 0,
            "min_precision": min(precisions) if precisions else 0,
            "max_precision": max(precisions) if precisions else 0,
            "hit_rate": sum(hits) / len(hits) if hits else 0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "num_queries": len(results),
        }


def load_chunks_from_dir(chunks_dir: Path) -> List[Chunk]:
    """Load all chunks from a directory of *_chunks.json files."""
    chunks = []
    for chunk_file in sorted(chunks_dir.glob("*_chunks.json")):
        with open(chunk_file) as f:
            chunks_data = json.load(f)
        for cd in chunks_data:
            chunks.append(Chunk(
                chunk_id=cd["chunk_id"],
                core_text=cd["core_text"],
                pages=cd["pages"],
                citation=cd["citation"],
                citation_string=cd["citation_string"],
                key_quotes=cd.get("key_quotes", []),
                tokens=cd.get("tokens", 0),
                doc_type=DocumentType(cd.get("doc_type", "unknown")),
            ))
    return chunks


def evaluate_query(
    query: BenchmarkQuery,
    bm25_indexer: BM25Indexer,
    chunks_by_id: Dict[str, Chunk],
    top_k: int,
    rerank: bool = False,
    reranker=None,
) -> QueryResult:
    """Run a single query and evaluate precision."""
    mode = "bm25+rerank" if rerank else "bm25"

    # Fetch more candidates when reranking
    fetch_k = top_k * 10 if rerank else top_k

    start = time.perf_counter()
    raw_results = bm25_indexer.search(query.query, top_k=fetch_k)
    bm25_elapsed = time.perf_counter() - start

    # Build SearchResult objects
    search_results = []
    for rank, (chunk_id, score) in enumerate(raw_results, 1):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        search_results.append(SearchResult(
            chunk_id=chunk_id,
            chunk=chunk,
            score=score,
            bm25_score=score,
            rank=rank,
        ))

    # Apply reranking if requested
    rerank_elapsed = 0.0
    if rerank and reranker is not None and search_results:
        start = time.perf_counter()
        search_results = reranker.rerank(query.query, search_results, top_k=top_k)
        rerank_elapsed = time.perf_counter() - start
    else:
        search_results = search_results[:top_k]

    total_ms = (bm25_elapsed + rerank_elapsed) * 1000

    # Compute precision: fraction of top-K results from expected documents
    hits = 0
    for sr in search_results:
        for prefix in query.expected_doc_prefixes:
            if sr.chunk_id.startswith(prefix):
                hits += 1
                break

    precision = hits / top_k if top_k > 0 else 0
    any_hit = hits > 0

    return QueryResult(
        query=query.query,
        description=query.description,
        mode=mode,
        top_k=top_k,
        precision=precision,
        latency_ms=total_ms,
        result_chunk_ids=[sr.chunk_id for sr in search_results],
        result_scores=[sr.score for sr in search_results],
        hit=any_hit,
    )


def run_benchmark(
    chunks_dir: Path,
    index_dir: Path,
    top_k: int = 5,
    rebuild: bool = False,
    queries: Optional[List[BenchmarkQuery]] = None,
) -> BenchmarkReport:
    """Run the full benchmark suite."""
    if queries is None:
        queries = BENCHMARK_QUERIES

    # Load chunks
    chunks = load_chunks_from_dir(chunks_dir)
    if not chunks:
        print(f"ERROR: No chunk files found in {chunks_dir}")
        sys.exit(1)

    chunks_by_id = {c.chunk_id: c for c in chunks}
    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")

    # Build or load BM25 index
    bm25 = BM25Indexer(index_dir=str(index_dir))
    if rebuild or not bm25.is_available():
        print("Building BM25 index...")
        start = time.perf_counter()
        bm25.build_index(chunks)
        build_ms = (time.perf_counter() - start) * 1000
        print(f"BM25 index built in {build_ms:.1f}ms ({len(chunks)} chunks)")
    else:
        bm25.load_index()
        print(f"Loaded existing BM25 index ({len(bm25.chunk_ids)} chunks)")

    # Initialize reranker
    reranker = None
    reranker_available = False
    try:
        from lit_pipeline.reranker import Reranker
        reranker = Reranker()
        reranker_available = reranker.is_available()
        if reranker_available:
            print("Cross-encoder reranker: AVAILABLE")
        else:
            print("Cross-encoder reranker: NOT AVAILABLE (model failed to load)")
    except ImportError:
        print("Cross-encoder reranker: NOT INSTALLED")

    report = BenchmarkReport(
        total_chunks=len(chunks),
        num_queries=len(queries),
        top_k=top_k,
    )

    # --- BM25-only ---
    print(f"\n{'='*70}")
    print(f"Running BM25-only benchmark ({len(queries)} queries, top-{top_k})")
    print(f"{'='*70}")

    bm25_results = []
    for q in queries:
        result = evaluate_query(q, bm25, chunks_by_id, top_k, rerank=False)
        bm25_results.append(result)
        report.query_results.append(result)

        status = "HIT" if result.hit else "MISS"
        print(f"  [{status}] P@{top_k}={result.precision:.2f}  {result.latency_ms:6.1f}ms  {q.query[:60]}")

    report.add_mode_summary("bm25", bm25_results)

    # --- BM25 + Rerank ---
    if reranker_available:
        print(f"\n{'='*70}")
        print(f"Running BM25+Rerank benchmark ({len(queries)} queries, top-{top_k})")
        print(f"{'='*70}")

        rerank_results = []
        for q in queries:
            result = evaluate_query(q, bm25, chunks_by_id, top_k, rerank=True, reranker=reranker)
            rerank_results.append(result)
            report.query_results.append(result)

            status = "HIT" if result.hit else "MISS"
            print(f"  [{status}] P@{top_k}={result.precision:.2f}  {result.latency_ms:6.1f}ms  {q.query[:60]}")

        report.add_mode_summary("bm25+rerank", rerank_results)

    return report


def print_report(report: BenchmarkReport):
    """Print a formatted benchmark report."""
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Corpus: {report.total_chunks} chunks | Queries: {report.num_queries} | top-K: {report.top_k}")
    print()

    # Mode comparison table
    header = f"{'Metric':<30} "
    modes = list(report.modes.keys())
    for mode in modes:
        header += f"{'  ' + mode:>16}"
    print(header)
    print("-" * (30 + 16 * len(modes)))

    metrics = [
        ("Mean Precision@K", "mean_precision", ".3f"),
        ("Min Precision@K", "min_precision", ".3f"),
        ("Max Precision@K", "max_precision", ".3f"),
        ("Hit Rate (any relevant)", "hit_rate", ".1%"),
        ("Mean Latency (ms)", "mean_latency_ms", ".1f"),
        ("P50 Latency (ms)", "p50_latency_ms", ".1f"),
        ("P95 Latency (ms)", "p95_latency_ms", ".1f"),
        ("Max Latency (ms)", "max_latency_ms", ".1f"),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<30} "
        for mode in modes:
            val = report.modes[mode][key]
            row += f"{val:>16{fmt}}"
        print(row)

    # Rerank improvement
    if "bm25" in report.modes and "bm25+rerank" in report.modes:
        bm25_p = report.modes["bm25"]["mean_precision"]
        rerank_p = report.modes["bm25+rerank"]["mean_precision"]
        delta = rerank_p - bm25_p
        bm25_lat = report.modes["bm25"]["mean_latency_ms"]
        rerank_lat = report.modes["bm25+rerank"]["mean_latency_ms"]
        overhead = rerank_lat - bm25_lat

        print()
        print(f"Rerank Impact:")
        print(f"  Precision improvement:  {delta:+.3f} ({bm25_p:.3f} → {rerank_p:.3f})")
        print(f"  Latency overhead:       {overhead:+.1f}ms ({bm25_lat:.1f}ms → {rerank_lat:.1f}ms)")
        print(f"  Hit rate change:        {report.modes['bm25']['hit_rate']:.1%} → {report.modes['bm25+rerank']['hit_rate']:.1%}")

    # Per-query detail for misses
    misses = [r for r in report.query_results if not r.hit]
    if misses:
        print(f"\nMISSED QUERIES ({len(misses)}):")
        for r in misses:
            print(f"  [{r.mode}] {r.query[:70]}")
            matching = [q for q in BENCHMARK_QUERIES if q.query == r.query]
            if matching:
                print(f"    Expected: {', '.join(matching[0].expected_doc_prefixes)}")
            if r.result_chunk_ids:
                print(f"    Got:      {r.result_chunk_ids[0]} (score={r.result_scores[0]:.3f})")

    # Per-query comparison between modes
    if len(modes) > 1:
        print(f"\nPER-QUERY COMPARISON (BM25 vs Rerank):")
        bm25_by_q = {r.query: r for r in report.query_results if r.mode == "bm25"}
        rerank_by_q = {r.query: r for r in report.query_results if r.mode == "bm25+rerank"}
        for q in BENCHMARK_QUERIES:
            b = bm25_by_q.get(q.query)
            rr = rerank_by_q.get(q.query)
            if b and rr:
                delta = rr.precision - b.precision
                if delta > 0:
                    marker = "^"
                elif delta < 0:
                    marker = "v"
                else:
                    marker = " "
                print(f"  {marker} {delta:+.2f}  BM25={b.precision:.2f} Rerank={rr.precision:.2f}  {q.query[:55]}")


def save_report_json(report: BenchmarkReport, output_path: Path):
    """Save benchmark report as JSON."""
    data = {
        "total_chunks": report.total_chunks,
        "num_queries": report.num_queries,
        "top_k": report.top_k,
        "modes": report.modes,
        "queries": [
            {
                "query": r.query,
                "description": r.description,
                "mode": r.mode,
                "precision": r.precision,
                "latency_ms": r.latency_ms,
                "hit": r.hit,
                "result_chunk_ids": r.result_chunk_ids,
                "result_scores": r.result_scores,
            }
            for r in report.query_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search quality and reranker performance"
    )
    parser.add_argument(
        "--chunks-dir",
        default="tests/pipeline_output/converted",
        help="Directory containing *_chunks.json files (default: tests/pipeline_output/converted)",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory for index files (default: temp directory)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="K for Precision@K evaluation (default: 5)",
    )
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Force rebuild of BM25 index",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        print(f"ERROR: Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    # Use temp dir for index if not specified
    temp_index = None
    if args.index_dir:
        index_dir = Path(args.index_dir)
    else:
        temp_index = tempfile.mkdtemp(prefix="benchmark_index_")
        index_dir = Path(temp_index)

    try:
        report = run_benchmark(
            chunks_dir=chunks_dir,
            index_dir=index_dir,
            top_k=args.top_k,
            rebuild=args.rebuild_index or temp_index is not None,
        )
        print_report(report)

        output_path = Path(args.output) if args.output else Path("benchmark_results.json")
        save_report_json(report, output_path)

    finally:
        if temp_index:
            shutil.rmtree(temp_index, ignore_errors=True)


if __name__ == "__main__":
    main()
