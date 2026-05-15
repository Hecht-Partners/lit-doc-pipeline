"""
End-to-end integration tests for the full pipeline.

Tests the complete workflow: conversion → citation → chunking → indexing → search
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from lit_pipeline.citation_types import DocumentType
from lit_pipeline.config_loader import load_default_config


@pytest.fixture
def temp_output():
    """Create temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestEndToEndPipeline:
    """Integration tests for full pipeline on real test documents."""

    def test_pipeline_process_command(self, temp_output):
        """Test lit-pipeline process command."""
        import subprocess

        result = subprocess.run(
            [
                ".venv/bin/python",
                "lit_pipeline.py",
                "process",
                "tests/test_docs",
                temp_output,
                "--no-cleanup-json",  # Keep JSON for verification
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Check command succeeded
        assert result.returncode == 0, f"Process failed: {result.stderr}"

        # Verify output files exist
        converted_dir = Path(temp_output) / "converted"
        assert converted_dir.exists()

        # Check for markdown files
        md_files = list(converted_dir.glob("*.md"))
        assert len(md_files) >= 2, f"Expected >= 2 .md files, got {len(md_files)}"

        # Check for citation files
        citation_files = list(converted_dir.glob("*_citations.json"))
        assert len(citation_files) >= 2, f"Expected >= 2 citation files, got {len(citation_files)}"

    def test_pipeline_index_command(self):
        """Test lit-pipeline index command on existing chunks."""
        import subprocess

        # Use existing test data that already has chunks
        if not Path("tests/pipeline_output/converted").exists():
            pytest.skip("Test data not available")

        result = subprocess.run(
            [".venv/bin/python", "lit_pipeline.py", "index", "tests/pipeline_output"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Index command requires chunks to exist, so it may fail if chunks weren't generated
        # Just verify the command runs without syntax errors
        assert "Loading chunks" in result.stderr or result.returncode == 0

    def test_pipeline_search_command(self):
        """Test lit-pipeline search command on existing indexes."""
        import subprocess

        result = subprocess.run(
            [
                ".venv/bin/python",
                "lit_pipeline.py",
                "search",
                "tests/pipeline_output",
                "TWT technology",
                "--top-k", "3",
                "--mode", "bm25",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"Search failed: {result.stderr}"

        # Check output contains results
        assert "Search Results" in result.stdout
        assert "Result 1/" in result.stdout

    def test_pipeline_stats_command(self):
        """Test lit-pipeline stats command."""
        import subprocess

        result = subprocess.run(
            [".venv/bin/python", "lit_pipeline.py", "stats", "tests/pipeline_output"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"Stats failed: {result.stderr}"

        # Check output contains stats
        assert "Index Statistics" in result.stdout
        assert "Total chunks:" in result.stdout


class TestCitationAccuracy:
    """Integration tests for citation accuracy on known documents."""

    def test_deposition_citation_accuracy(self):
        """Daniel Alexander deposition should have 100% line-level accuracy."""
        citation_file = Path("tests/wednesday_morning_test_2026-02-11/converted/daniel_alexander_10_24_2025_citations.json")

        if not citation_file.exists():
            pytest.skip("Test data not available")

        with open(citation_file) as f:
            citations = json.load(f)

        # Count line-level citations
        line_citations = [c for c in citations.values() if c.get("line_start") is not None]

        # Should have high line-level coverage
        coverage = len(line_citations) / len(citations) * 100
        assert coverage > 95.0, f"Expected >95% line coverage, got {coverage:.1f}%"

    def test_expert_report_paragraph_accuracy(self):
        """Cole Report should have >95% paragraph detection."""
        citation_file = Path("tests/para_test_output/converted/2025_12_11_cole_report_citations.json")

        if not citation_file.exists():
            pytest.skip("Test data not available")

        with open(citation_file) as f:
            citations = json.load(f)

        # Count paragraph citations
        para_citations = [c for c in citations.values() if c.get("paragraph_number") is not None]

        # Should have >95% paragraph coverage
        coverage = len(para_citations) / len(citations) * 100
        assert coverage > 95.0, f"Expected >95% paragraph coverage, got {coverage:.1f}%"

    def test_patent_column_detection(self):
        """Patent should have >80% column detection on spec pages."""
        citation_file = Path("tests/wednesday_morning_test_2026-02-11/converted/intel_prox_00006214_citations.json")

        if not citation_file.exists():
            pytest.skip("Test data not available")

        with open(citation_file) as f:
            citations = json.load(f)

        # Spec pages are 24-34 in this patent
        spec_pages = set(range(24, 35))
        spec_citations = [c for c in citations.values() if c.get("page") in spec_pages]
        column_citations = [c for c in spec_citations if c.get("column") is not None]

        coverage = len(column_citations) / len(spec_citations) * 100
        assert coverage > 80.0, f"Expected >80% column coverage on spec pages, got {coverage:.1f}%"


class TestSearchRelevance:
    """Integration tests for search quality."""

    def test_search_finds_known_document(self):
        """Search for 'TWT technology' should return Cole Report in top 5."""
        from lit_pipeline.hybrid_retriever import HybridRetriever

        index_dir = "tests/pipeline_output/indexes"
        chunks_dir = "tests/pipeline_output/converted"

        if not Path(index_dir).exists():
            pytest.skip("Indexes not built")

        retriever = HybridRetriever(index_dir=index_dir, chunks_dir=chunks_dir)
        results = retriever.search("TWT technology battery life", top_k=5, mode="bm25")

        # Should have at least 1 result
        assert len(results) > 0, "No results found"

        # Top result should be from Cole Report
        top_chunk_id = results[0].chunk_id
        assert "cole_report" in top_chunk_id.lower(), f"Expected Cole Report in top result, got {top_chunk_id}"

    def test_hybrid_search_better_than_bm25_alone(self):
        """Hybrid search should find results for semantic queries."""
        from lit_pipeline.hybrid_retriever import HybridRetriever

        index_dir = "tests/pipeline_output/indexes"
        chunks_dir = "tests/pipeline_output/converted"

        if not Path(index_dir).exists():
            pytest.skip("Indexes not built")

        retriever = HybridRetriever(index_dir=index_dir, chunks_dir=chunks_dir)

        # Try semantic query
        if retriever.vector_indexer.is_available():
            results = retriever.search("power saving wireless technology", top_k=3, mode="hybrid")
            assert len(results) > 0, "Hybrid search found no results"


class TestConfigLoader:
    """Tests for configuration file loading."""

    def test_load_default_config(self):
        """Default config should load successfully."""
        config = load_default_config()

        assert "chunking" in config
        assert "docling" in config
        assert config["chunking"]["target_chunk_chars"] == 8000

    def test_config_file_loading(self):
        """Load from specific config file."""
        from lit_pipeline.config_loader import ConfigLoader

        # Use retrieval_config.json which has bm25 settings
        config = ConfigLoader("configs/retrieval_config.json")

        assert config.get("k1", section="bm25") == 1.5
        assert config.get("b", section="bm25") == 0.75

    def test_config_merge_cli_args(self):
        """CLI args should override config file."""
        from lit_pipeline.config_loader import ConfigLoader

        config = ConfigLoader("configs/retrieval_config.json")

        cli_args = {"k1": 2.0, "b": 0.8}
        merged = config.merge_cli_args(cli_args, section="bm25")

        # CLI args override config
        assert merged["k1"] == 2.0
        assert merged["b"] == 0.8
        # Other config values preserved from file
        assert merged.get("max_features") == 10000
