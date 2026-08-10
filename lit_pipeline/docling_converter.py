"""
Docling-based document converter with inline citation extraction.

Strategy: Extract citations DURING conversion from markdown output,
NOT from JSON (JSON wastes space with garbage text from image processing).
"""

import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from lit_pipeline.citation_types import ConversionResult, DocumentType


class DoclingConverter:
    """
    Convert documents to markdown, extracting citation markers inline.

    CRITICAL: Avoid JSON output (wastes space with garbage text).
    Extract citations directly from conversion process.
    """

    def __init__(self, config: Optional[Dict] = None, timeout: int = 300):
        """
        Initialize converter with configuration.

        Args:
            config: Configuration dict with docling settings
            timeout: Timeout in seconds for document conversion (default: 300)
        """
        self.config = config or self._default_config()
        self.timeout = timeout

        # Citation patterns
        self.page_pattern = re.compile(r'(?:Page|p\.)\s+(\d+)', re.IGNORECASE)
        self.bates_patterns = [
            re.compile(r'INTEL_PROX_\d{5,11}'),
            re.compile(r'PROX_INTEL-\d{5,11}'),
            re.compile(r'[A-Z]{2,}[-_][A-Z]{2,}[-_]\d{5,}'),
            # Separator-less production stamps (e.g. ABBOTT0000123)
            re.compile(r'\b[A-Z]{2,10}\d{6,10}\b'),
        ]
        # Deposition line numbers (Q/A format with line numbers)
        self.line_pattern = re.compile(r'^\s*(\d{1,2})\s+[QA]\s+', re.MULTILINE)
        # Column markers
        self.col_pattern = re.compile(r'col(?:umn)?\.?\s*(\d+)', re.IGNORECASE)
        # Paragraph markers
        self.para_pattern = re.compile(r'[¶§]\s*(\d+)')

    def _default_config(self) -> Dict:
        """Default Docling configuration."""
        return {
            "image_export_mode": "placeholder",
            "enrich_picture_classes": False,
            "enrich_picture_description": False,
            "enrich_chart_extraction": False,
            "enable_ocr": True
        }

    def convert_document(self, input_path: str, output_dir: str) -> ConversionResult:
        """
        Convert document with inline citation markers.

        Strategy:
        1. Use Docling with --to md only (no JSON)
        2. Parse markdown output for citation patterns:
           - Page markers: "Page 14", "p. 14"
           - Line numbers: "1", "2", ... "25" (in deposition context)
           - Bates stamps: "INTEL_PROX_00001770"
           - Column markers: "col. 3", "column 4"
           - Paragraph markers: "¶ 42", "Paragraph 42"
        3. Add structured markers in markdown: [PAGE:14], [LINE:5], [BATES:...]
        4. Save enhanced markdown with embedded citation metadata

        Args:
            input_path: Path to input document
            output_dir: Directory to save converted files

        Returns:
            ConversionResult with paths and citation data
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build Docling command
        cmd = self._build_docling_command(input_path, output_dir)

        # Run conversion
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                error_msg = f"Docling conversion failed: {result.stderr}"
                return ConversionResult(
                    md_path="",
                    citations_found={},
                    needs_reconstruction=True,
                    errors=[error_msg]
                )

        except subprocess.TimeoutExpired:
            return ConversionResult(
                md_path="",
                citations_found={},
                needs_reconstruction=True,
                errors=[f"Conversion timeout (>{self.timeout} seconds)"]
            )
        except Exception as e:
            return ConversionResult(
                md_path="",
                citations_found={},
                needs_reconstruction=True,
                errors=[f"Conversion error: {str(e)}"]
            )

        # Find output markdown file
        md_path = output_dir / f"{input_path.stem}.md"
        if not md_path.exists():
            return ConversionResult(
                md_path="",
                citations_found={},
                needs_reconstruction=True,
                errors=[f"Expected output not found: {md_path}"]
            )

        # Extract citations from markdown
        citations = self._extract_citations_from_markdown(md_path)

        # Detect document type
        doc_type = self._detect_document_type(citations)

        # Assess whether reconstruction is needed
        needs_reconstruction = self._assess_completeness(citations, doc_type)

        return ConversionResult(
            md_path=str(md_path),
            citations_found=citations,
            needs_reconstruction=needs_reconstruction,
            doc_type=doc_type
        )

    def _build_docling_command(self, input_path: Path, output_dir: Path) -> List[str]:
        """Build Docling command with proper flags."""
        import sys
        # Use docling from the same venv as the running Python interpreter
        docling_bin = str(Path(sys.executable).parent / "docling")
        cmd = [
            docling_bin,
            "--to", "md", "--to", "json",  # Both markdown and JSON for citation tracking
            "--image-export-mode", self.config["image_export_mode"],
            "--pdf-backend", "pypdfium2",  # Try pypdfium2 backend for better text formatting extraction
            "--output", str(output_dir),
            str(input_path)
        ]

        # Add flag-based options
        if not self.config["enrich_picture_classes"]:
            cmd.append("--no-enrich-picture-classes")
        if not self.config["enrich_picture_description"]:
            cmd.append("--no-enrich-picture-description")
        if not self.config["enrich_chart_extraction"]:
            cmd.append("--no-enrich-chart-extraction")
        if not self.config["enable_ocr"]:
            cmd.append("--no-ocr")

        return cmd

    def _extract_citations_from_markdown(self, md_path: Path) -> Dict[str, List]:
        """
        Extract citation patterns from markdown text.

        Patterns to detect:
        - Page breaks: "---\nPage 14\n---"
        - Line numbers: "^\s*(\d{1,2})\s+[QA]\s+" (deposition format)
        - Bates: "INTEL_PROX_\d{5,11}"
        - Columns: "col\. (\d+)"
        - Paragraphs: "[¶§]\s*(\d+)"

        Returns:
            Dictionary with lists of found citations by type
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        citations = {
            'pages': [],
            'line_markers': [],
            'bates_stamps': [],
            'column_markers': [],
            'paragraph_markers': []
        }

        # Extract page markers
        page_matches = self.page_pattern.findall(content)
        citations['pages'] = [int(m) for m in page_matches]

        # Extract Bates stamps
        for pattern in self.bates_patterns:
            citations['bates_stamps'].extend(pattern.findall(content))
        # Remove duplicates while preserving order
        citations['bates_stamps'] = list(dict.fromkeys(citations['bates_stamps']))

        # Deposition line numbers (Q/A format with line numbers)
        line_matches = self.line_pattern.findall(content)
        citations['line_markers'] = [int(m) for m in line_matches]

        # Column markers
        col_matches = self.col_pattern.findall(content)
        citations['column_markers'] = [int(m) for m in col_matches]

        # Paragraph markers
        para_matches = self.para_pattern.findall(content)
        citations['paragraph_markers'] = [int(m) for m in para_matches]

        return citations

    def _detect_document_type(self, citations: Dict[str, List]) -> DocumentType:
        """
        Detect document type based on citation patterns.

        Heuristics:
        - Deposition: Has line markers (1-25) and Q/A format
        - Patent: Has column markers
        - Expert Report: Has paragraph markers (¶)
        - Pleading: Has numbered paragraphs but no Q/A
        """
        has_lines = len(citations.get('line_markers', [])) > 0
        has_columns = len(citations.get('column_markers', [])) > 0
        has_paragraphs = len(citations.get('paragraph_markers', [])) > 0

        if has_lines:
            return DocumentType.DEPOSITION
        elif has_columns:
            return DocumentType.PATENT
        elif has_paragraphs:
            return DocumentType.EXPERT_REPORT
        else:
            return DocumentType.UNKNOWN

    def _assess_completeness(self, citations: Dict[str, List], doc_type: DocumentType) -> bool:
        """
        Assess whether citation reconstruction is needed.

        Returns:
            True if reconstruction script is needed, False if citations are sufficient
        """
        # Check if we have pages (baseline requirement)
        if not citations.get('pages'):
            return True  # Need reconstruction if no pages found

        # Type-specific checks
        if doc_type == DocumentType.DEPOSITION:
            # Depositions need line markers
            if not citations.get('line_markers'):
                return True

        elif doc_type == DocumentType.PATENT:
            # Patents need column markers
            if not citations.get('column_markers'):
                return True

        elif doc_type == DocumentType.EXPERT_REPORT:
            # Expert reports need paragraph markers
            if not citations.get('paragraph_markers'):
                return True

        # If we got here, citations look sufficient
        return False


def main():
    """Test converter on a sample document."""
    import sys
    if len(sys.argv) < 3:
        print("Usage: python docling_converter.py <input_file> <output_dir>")
        sys.exit(1)

    converter = DoclingConverter()
    result = converter.convert_document(sys.argv[1], sys.argv[2])

    print(result.citation_coverage_summary())

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
