#!/usr/bin/env python3
"""
Quick test script for Phase 1 conversion.

Usage:
    python test_conversion.py document.pdf
    python test_conversion.py document.xlsx
"""

import sys
from pathlib import Path
from lit_pipeline.docling_converter import DoclingConverter
from lit_pipeline.format_handlers import FormatHandler
from lit_pipeline.post_processor import PostProcessor


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_conversion.py <document_path>")
        print("\nSupported formats: PDF, DOCX, XLSX, XLS, EML, MSG, PPTX, TXT, MD")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing Conversion: {input_path.name}")
    print(f"{'='*60}\n")

    # Choose converter based on file type
    if input_path.suffix.lower() in ['.pdf', '.docx']:
        print("Using Docling converter...")
        converter = DoclingConverter()
        result = converter.convert_document(str(input_path), str(output_dir))
    else:
        print(f"Using format handler for {input_path.suffix}...")
        handler = FormatHandler(str(output_dir))
        result = handler.convert(input_path)

    # Check for errors
    if result.errors:
        print("\n❌ Conversion failed:")
        for error in result.errors:
            print(f"  {error}")
        sys.exit(1)

    # Print results
    print("\n✅ Conversion successful!")
    print(result.citation_coverage_summary())

    # Post-process if conversion succeeded
    if result.md_path:
        print(f"\n{'='*60}")
        print("Post-Processing")
        print(f"{'='*60}\n")

        processor = PostProcessor()
        proc_result = processor.process(result.md_path, result.doc_type)

        print("\n✅ Post-processing complete!")
        print(proc_result)

        print(f"\n{'='*60}")
        print("Output Files")
        print(f"{'='*60}")
        print(f"  Markdown: {proc_result.cleaned_path}")
        print(f"  Citations: {proc_result.citations_path}")
        print(f"\nYou can now review these files to assess citation extraction quality.")

    print()


if __name__ == "__main__":
    main()
