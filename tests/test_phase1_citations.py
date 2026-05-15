"""
Phase 1 Testing: Citation Extraction Coverage

Tests to assess what citations we successfully extracted from Phase 1
(conversion + post-processing) and identify gaps for Phase 2 reconstruction.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lit_pipeline.docling_converter import DoclingConverter
from lit_pipeline.post_processor import PostProcessor
from lit_pipeline.format_handlers import FormatHandler
from lit_pipeline.citation_types import DocumentType


class CitationCoverageTester:
    """Test citation extraction coverage from Phase 1."""

    def __init__(self):
        self.converter = DoclingConverter()
        self.processor = PostProcessor()

    def test_document(self, input_path: str, output_dir: str) -> Dict:
        """
        Test citation extraction on a document.

        Returns:
            Coverage report with statistics and recommendations
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Testing: {input_path.name}")
        print(f"{'='*60}\n")

        # Step 1: Convert document
        print("Step 1: Converting document...")
        if input_path.suffix.lower() in ['.pdf', '.docx']:
            conv_result = self.converter.convert_document(str(input_path), str(output_dir))
        else:
            handler = FormatHandler(str(output_dir))
            conv_result = handler.convert(input_path)

        if conv_result.errors:
            print("❌ Conversion failed:")
            for error in conv_result.errors:
                print(f"  {error}")
            return self._failed_report(conv_result.errors)

        print(conv_result.citation_coverage_summary())

        # Step 2: Post-process
        print("\nStep 2: Post-processing...")
        proc_result = self.processor.process(
            conv_result.md_path,
            conv_result.doc_type
        )
        print(proc_result)

        # Step 3: Analyze coverage
        print("\nStep 3: Analyzing citation coverage...")
        coverage = self._analyze_coverage(
            conv_result.citations_found,
            conv_result.doc_type,
            proc_result.citations_path
        )

        # Step 4: Generate recommendations
        print("\nStep 4: Generating recommendations...")
        recommendations = self._generate_recommendations(coverage, conv_result.doc_type)

        # Print report
        self._print_report(coverage, recommendations)

        return {
            "coverage": coverage,
            "recommendations": recommendations,
            "doc_type": conv_result.doc_type.value,
            "needs_reconstruction": conv_result.needs_reconstruction
        }

    def _analyze_coverage(
        self,
        citations: Dict[str, List],
        doc_type: DocumentType,
        citations_path: str
    ) -> Dict:
        """
        Analyze citation coverage.

        Returns:
            Coverage statistics
        """
        # Load citation map from post-processing
        with open(citations_path, 'r') as f:
            citation_map = json.load(f)

        coverage = {
            "pages": {
                "found": len(citations.get('pages', [])),
                "coverage": "✅ Good" if citations.get('pages') else "❌ None"
            },
            "bates": {
                "found": len(citations.get('bates_stamps', [])),
                "coverage": "✅ Good" if citations.get('bates_stamps') else "⚠️ None"
            },
            "citation_map_entries": len(citation_map)
        }

        # Type-specific coverage
        if doc_type == DocumentType.DEPOSITION:
            line_count = len(citations.get('line_markers', []))
            coverage["line_numbers"] = {
                "found": line_count,
                "coverage": "✅ Good" if line_count > 20 else "❌ Low" if line_count > 0 else "❌ None"
            }

        elif doc_type == DocumentType.PATENT:
            col_count = len(citations.get('column_markers', []))
            coverage["column_markers"] = {
                "found": col_count,
                "coverage": "✅ Good" if col_count > 0 else "❌ None"
            }

        elif doc_type == DocumentType.EXPERT_REPORT:
            para_count = len(citations.get('paragraph_markers', []))
            coverage["paragraph_markers"] = {
                "found": para_count,
                "coverage": "✅ Good" if para_count > 5 else "❌ Low" if para_count > 0 else "❌ None"
            }

        return coverage

    def _generate_recommendations(self, coverage: Dict, doc_type: DocumentType) -> List[str]:
        """
        Generate recommendations based on coverage.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check page coverage
        if coverage["pages"]["found"] == 0:
            recommendations.append(
                "⚠️ No page markers found. Consider improving page detection in converter."
            )
        else:
            recommendations.append(
                f"✅ Found {coverage['pages']['found']} page markers - good coverage."
            )

        # Check Bates coverage
        if coverage["bates"]["found"] == 0:
            recommendations.append(
                "⚠️ No Bates stamps found. If document has Bates, improve pattern matching."
            )

        # Type-specific recommendations
        if doc_type == DocumentType.DEPOSITION:
            line_count = coverage.get("line_numbers", {}).get("found", 0)
            if line_count == 0:
                recommendations.append(
                    "❌ No line numbers found for deposition. CRITICAL: Need citation reconstruction script."
                )
            elif line_count < 20:
                recommendations.append(
                    f"⚠️ Only {line_count} line markers found. May need reconstruction to improve coverage."
                )
            else:
                recommendations.append(
                    f"✅ Found {line_count} line markers - good coverage for deposition."
                )

        elif doc_type == DocumentType.PATENT:
            col_count = coverage.get("column_markers", {}).get("found", 0)
            if col_count == 0:
                recommendations.append(
                    "❌ No column markers found for patent. Need citation reconstruction script."
                )
            else:
                recommendations.append(
                    f"✅ Found {col_count} column markers - good coverage for patent."
                )

        elif doc_type == DocumentType.EXPERT_REPORT:
            para_count = coverage.get("paragraph_markers", {}).get("found", 0)
            if para_count == 0:
                recommendations.append(
                    "❌ No paragraph markers found. Need citation reconstruction script."
                )
            elif para_count < 5:
                recommendations.append(
                    f"⚠️ Only {para_count} paragraph markers found. May need reconstruction."
                )
            else:
                recommendations.append(
                    f"✅ Found {para_count} paragraph markers - good coverage."
                )

        # Overall assessment
        citation_entries = coverage.get("citation_map_entries", 0)
        if citation_entries == 0:
            recommendations.append(
                "\n❌ BLOCKER: No citation map entries created. Cannot proceed to chunking without citations."
            )
        elif citation_entries < 10:
            recommendations.append(
                f"\n⚠️ Only {citation_entries} citation entries. Consider improving Phase 1 extraction."
            )
        else:
            recommendations.append(
                f"\n✅ Created {citation_entries} citation entries. Ready for chunking or may proceed with reconstruction."
            )

        return recommendations

    def _print_report(self, coverage: Dict, recommendations: List[str]):
        """Print coverage report."""
        print("\n" + "="*60)
        print("CITATION COVERAGE REPORT")
        print("="*60)

        print("\nCoverage Statistics:")
        for key, value in coverage.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                print(f"    Found: {value.get('found', 0)}")
                print(f"    Status: {value.get('coverage', 'Unknown')}")
            else:
                print(f"  {key}: {value}")

        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "="*60)

    def _failed_report(self, errors: List[str]) -> Dict:
        """Generate report for failed conversion."""
        return {
            "coverage": {},
            "recommendations": [
                "❌ Conversion failed. Cannot assess citation coverage.",
                *[f"  Error: {e}" for e in errors]
            ],
            "doc_type": "unknown",
            "needs_reconstruction": True
        }


def main():
    """Run citation coverage tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test citation extraction coverage from Phase 1"
    )
    parser.add_argument("input", help="Input document path")
    parser.add_argument("--output-dir", default="./test_output",
                       help="Output directory for converted files")

    args = parser.parse_args()

    tester = CitationCoverageTester()
    result = tester.test_document(args.input, args.output_dir)

    # Exit with error code if reconstruction is needed
    if result.get("needs_reconstruction"):
        print("\n⚠️ Citation reconstruction will be needed for this document type.")
        sys.exit(0)  # Not a failure, just informational
    else:
        print("\n✅ Citation coverage is sufficient. Can proceed to chunking.")
        sys.exit(0)


if __name__ == "__main__":
    main()
