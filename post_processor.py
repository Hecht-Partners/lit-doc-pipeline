"""
Post-processor for cleaning markdown and enhancing citation markers.

Strategy:
1. Parse markdown structure
2. Identify sections with citation context
3. Insert structured citation markers
4. Create citation map: text_position → citation_data
5. Move footnotes inline with their paragraphs
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from citation_types import ProcessingResult, CitationData, DocumentType

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Clean markdown while enhancing citation markers.

    CRITICAL: Must preserve citation information while cleaning text.
    Never strip line numbers, page markers, or Bates stamps before
    associating them with text content.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize post-processor.

        Args:
            config: Configuration dict with bates patterns, etc.
        """
        self.config = config or {}
        self.bates_patterns = self._load_bates_patterns()

    def _load_bates_patterns(self) -> List[re.Pattern]:
        """Load Bates stamp patterns from config."""
        pattern_strs = self.config.get('bates_patterns', [
            r'INTEL_PROX_\d{5,11}',
            r'PROX_INTEL-\d{5,11}',
            r'[A-Z]{2,}[-_][A-Z]{2,}[-_]\d{5,}'
        ])
        return [re.compile(p) for p in pattern_strs]

    def process(self, md_path: str, doc_type: DocumentType = DocumentType.UNKNOWN) -> ProcessingResult:
        """
        Clean markdown and enhance with structured citation markers.

        Args:
            md_path: Path to markdown file from conversion
            doc_type: Detected document type

        Returns:
            ProcessingResult with paths and coverage info
        """
        md_path = Path(md_path)

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Load Docling JSON for footnote detection and text markers
        footnotes = []
        text_elements = []
        json_data = self._load_docling_json(md_path)

        if json_data:
            # Extract footnotes for inline insertion (all document types)
            # Footnotes appear in expert reports, pleadings, briefs, court opinions, etc.
            footnotes = self._extract_footnotes(json_data)
            if footnotes:
                logger.info("Detected %d footnotes for inline insertion", len(footnotes))

            # Extract text elements for marker insertion (all Docling docs)
            text_elements = self._extract_text_elements(json_data)
            logger.info("Extracted %d text elements for marker insertion", len(text_elements))

        # Clean content
        cleaned = self._clean_markdown(content)

        # Insert text markers for citation linkage (before footnote processing)
        if text_elements:
            cleaned = self._insert_text_markers(cleaned, text_elements)
            logger.info("Inserted %d text markers for citation linkage", len(text_elements))

        # Move footnotes inline (after text markers, before citation markers)
        if footnotes:
            cleaned = self._inline_footnotes(cleaned, footnotes)
            logger.info("Inserted %d footnotes inline", len(footnotes))

        # Enhance with structured citation markers
        enhanced, citation_map = self._add_citation_markers(cleaned, doc_type)

        # Save enhanced markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(enhanced)

        # Save citation map
        citation_path = str(md_path).replace('.md', '_citations.json')
        with open(citation_path, 'w', encoding='utf-8') as f:
            json.dump(citation_map, f, indent=2)

        return ProcessingResult(
            cleaned_path=str(md_path),
            citations_path=citation_path,
            citation_coverage=len(citation_map)
        )

    def _clean_markdown(self, content: str) -> str:
        """
        Clean markdown content while preserving citation markers.

        Removes:
        - Concordances (alphabetical word lists)
        - Table of contents
        - OCR artifacts (excessive whitespace, garbled characters)
        - Boilerplate headers/footers

        Preserves:
        - Line numbers (1-25 in depositions)
        - Page markers
        - Bates stamps
        - Q/A structure
        - Paragraph numbers
        """
        lines = content.split('\n')
        cleaned_lines = []

        # Track state
        in_concordance = False
        in_toc = False

        for line_idx, line in enumerate(lines):
            # Detect concordance sections (alphabetical word lists)
            if re.match(r'^[A-Z]\s*$', line) and len(cleaned_lines) > 0:
                # Single letter lines might indicate concordance
                if self._is_concordance_section(lines, line_idx):
                    in_concordance = True
                    continue

            # Detect table of contents
            if re.match(r'^(table\s+of\s+contents|contents)', line, re.IGNORECASE):
                in_toc = True
                continue

            # Exit TOC when we hit substantive content
            if in_toc and (
                re.match(r'^\d{1,2}\s+[QA]\s+', line) or  # Deposition Q/A
                re.match(r'^[¶§]\s*\d+', line) or  # Paragraph marker
                re.match(r'^CLAIM \d+', line, re.IGNORECASE)  # Patent claim
            ):
                in_toc = False

            # Skip if in concordance or TOC
            if in_concordance or in_toc:
                continue

            # Skip excessive whitespace
            if re.match(r'^\s*$', line) and (
                not cleaned_lines or re.match(r'^\s*$', cleaned_lines[-1])
            ):
                continue

            # Skip OCR artifacts (lines with excessive underscores or dots)
            if re.match(r'^[_\.\s]{10,}$', line):
                continue

            # Clean excessive whitespace within line
            line = re.sub(r'\s{3,}', '  ', line)

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _is_concordance_section(self, lines: List[str], start_idx: int) -> bool:
        """
        Detect if we're at the start of a concordance section.

        Concordances are alphabetical word lists, typically at end of document.
        Heuristic: Next 5-10 lines are single words in alphabetical order.
        """
        if start_idx + 10 >= len(lines):
            return False

        words = []
        for i in range(start_idx, min(start_idx + 10, len(lines))):
            line = lines[i].strip()
            if re.match(r'^[A-Za-z]+$', line):
                words.append(line.lower())
            else:
                break

        # Check if words are in alphabetical order
        if len(words) >= 5:
            return words == sorted(words)

        return False

    def _load_docling_json(self, md_path: Path) -> Optional[dict]:
        """
        Load the Docling JSON file corresponding to the markdown file.

        Args:
            md_path: Path to markdown file

        Returns:
            Parsed JSON data, or None if file not found
        """
        json_path = md_path.with_suffix('.json')
        if not json_path.exists():
            logger.debug("No JSON file found at %s", json_path)
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load JSON from %s: %s", json_path, e)
            return None

    def _extract_footnotes(self, json_data: dict) -> List[Dict[str, str]]:
        """
        Extract footnote elements from Docling JSON.

        Args:
            json_data: Parsed Docling JSON

        Returns:
            List of dicts with 'number' and 'text' keys
        """
        texts = json_data.get('texts', [])
        footnotes = []

        for elem in texts:
            if elem.get('label') == 'footnote':
                text = elem.get('text', '').strip()
                if not text:
                    continue

                # Parse footnote number from start of text (e.g., "1 See..." or "2 Id.")
                match = re.match(r'^(\d+)\s+(.+)$', text, re.DOTALL)
                if match:
                    num = match.group(1)
                    content = match.group(2).strip()
                    footnotes.append({
                        'number': num,
                        'text': text,  # Keep full text for matching
                        'content': content  # Content without number
                    })

        return footnotes

    def _extract_text_elements(self, json_data: dict) -> List[Dict[str, any]]:
        """
        Extract text elements from Docling JSON for marker insertion.

        Args:
            json_data: Parsed Docling JSON

        Returns:
            List of dicts with 'index', 'self_ref', 'text', and 'label' keys
        """
        texts = json_data.get('texts', [])
        elements = []

        for i, elem in enumerate(texts):
            text = elem.get('text', '').strip()
            if not text:
                continue

            # Skip footnotes (handled separately)
            if elem.get('label') == 'footnote':
                continue

            self_ref = elem.get('self_ref', f'#/texts/{i}')
            # Extract numeric index from self_ref (e.g., "#/texts/42" -> "42")
            match = re.search(r'/texts/(\d+)', self_ref)
            text_id = match.group(1) if match else str(i)

            elements.append({
                'index': i,
                'text_id': text_id,
                'self_ref': self_ref,
                'text': text,
                'label': elem.get('label', '')
            })

        return elements

    def _insert_text_markers(self, content: str, text_elements: List[Dict[str, any]]) -> str:
        """
        Insert [TEXT:N] markers in markdown for citation linkage.

        Matches markdown lines to JSON text elements and inserts markers
        like [TEXT:42] that can be used to look up citation metadata.

        Args:
            content: Markdown content
            text_elements: List of text element dicts from JSON

        Returns:
            Modified markdown with text markers
        """
        if not text_elements:
            return content

        lines = content.split('\n')
        result = []
        used_elements = set()  # Track which elements we've already marked

        for line in lines:
            line_stripped = line.strip()

            # Skip empty lines and headers
            if not line_stripped or line_stripped.startswith('#'):
                result.append(line)
                continue

            # Try to match this line to a text element
            matched_element = None
            for elem in text_elements:
                if elem['index'] in used_elements:
                    continue

                elem_text = elem['text']

                # Match if line contains the start of the element text
                # Use first 50 chars for matching to handle multi-line elements
                elem_prefix = elem_text[:min(50, len(elem_text))].strip()

                if elem_prefix and (
                    line_stripped.startswith(elem_prefix) or
                    elem_prefix in line_stripped or
                    # Handle tables and structured content
                    (len(elem_prefix) > 20 and elem_prefix[:20] in line_stripped)
                ):
                    matched_element = elem
                    used_elements.add(elem['index'])
                    break

            # Insert marker before matched lines
            if matched_element:
                result.append(f"[TEXT:{matched_element['text_id']}]")

            result.append(line)

        return '\n'.join(result)

    def _inline_footnotes(self, content: str, footnotes: List[Dict[str, str]]) -> str:
        """
        Move footnotes inline with their paragraphs in the markdown.

        Strategy:
        1. Identify footnote lines in markdown by number (more robust than text matching)
        2. Buffer footnotes and remove them from their current position
        3. Insert buffered footnotes at the end of the preceding paragraph

        Args:
            content: Markdown content
            footnotes: List of footnote dicts from JSON

        Returns:
            Modified markdown with footnotes inline
        """
        if not footnotes:
            return content

        lines = content.split('\n')
        result = []
        footnote_buffer = []

        # Build a dict of footnote number -> footnote data
        footnote_by_num = {fn['number']: fn for fn in footnotes}

        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()

            # Check if this line is a footnote by matching the number at the start
            is_footnote = False
            matched_fn = None

            # Match pattern: "1 " or "12 " at start of line (number + space)
            match = re.match(r'^(\d+)\s+', line_stripped)
            if match:
                fn_num = match.group(1)
                if fn_num in footnote_by_num:
                    # Verify this looks like a footnote (not just a numbered list)
                    # Footnotes typically have citations or "Id." or start with uppercase
                    rest = line_stripped[match.end():].strip()
                    if (rest.startswith('Id') or
                        rest.startswith('See') or
                        re.match(r'^[A-Z_]+', rest) or  # Starts with caps/underscores (Bates)
                        'at' in rest[:30]):  # Contains "at" (citation format)
                        matched_fn = footnote_by_num[fn_num]
                        is_footnote = True

            if is_footnote and matched_fn:
                # Buffer this footnote instead of adding to result
                footnote_buffer.append(matched_fn)
                i += 1
                continue

            # Check if we need to flush buffered footnotes
            # Flush when we hit: substantive text, new paragraph, section header
            should_flush = (
                footnote_buffer and
                line_stripped and  # Non-empty line
                not line_stripped.startswith('#') and  # Not a header continuation
                not re.match(r'^[\|\-\s]+$', line_stripped)  # Not table separator
            )

            if should_flush and result:
                # Insert footnotes after the last non-empty line in result
                # Find the last paragraph line
                insert_pos = len(result)
                while insert_pos > 0 and not result[insert_pos - 1].strip():
                    insert_pos -= 1

                # Insert footnotes
                for fn in footnote_buffer:
                    footnote_line = f"[FOOTNOTE {fn['number']}: {fn['content']}]"
                    result.insert(insert_pos, footnote_line)
                    insert_pos += 1

                footnote_buffer = []

            result.append(line)
            i += 1

        # Flush any remaining buffered footnotes at the end
        if footnote_buffer:
            for fn in footnote_buffer:
                footnote_line = f"[FOOTNOTE {fn['number']}: {fn['content']}]"
                result.append(footnote_line)

        return '\n'.join(result)

    def _add_citation_markers(
        self,
        content: str,
        doc_type: DocumentType
    ) -> Tuple[str, Dict[str, Dict]]:
        """
        Add structured citation markers to markdown.

        Strategy:
        1. Parse content for citation elements (pages, lines, Bates, etc.)
        2. Insert structured markers: [PAGE:14], [LINE:5], [BATES:...]
        3. Build citation map: line_index → CitationData

        Args:
            content: Cleaned markdown content
            doc_type: Detected document type

        Returns:
            Tuple of (enhanced_content, citation_map)
        """
        lines = content.split('\n')
        enhanced_lines = []
        citation_map = {}

        current_page = None
        current_bates = None
        current_line_num = None

        transcript_types = {DocumentType.DEPOSITION, DocumentType.HEARING_TRANSCRIPT}
        patent_types = {DocumentType.PATENT}

        for idx, line in enumerate(lines):
            # Detect page markers
            page_match = re.search(r'(?:Page|p\.)\s+(\d+)', line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                enhanced_lines.append(f"[PAGE:{current_page}]")
                enhanced_lines.append(line)
                continue

            # Detect Bates stamps
            bates_match = None
            for pattern in self.bates_patterns:
                bates_match = pattern.search(line)
                if bates_match:
                    current_bates = bates_match.group(0)
                    enhanced_lines.append(f"[BATES:{current_bates}]")
                    break

            # Detect deposition line numbers
            if doc_type in transcript_types:
                line_match = re.match(r'^\s*(\d{1,2})\s+([QA])\s+(.*)$', line)
                if line_match:
                    current_line_num = int(line_match.group(1))
                    qa_marker = line_match.group(2)
                    text = line_match.group(3)

                    # Create citation entry
                    citation_data = CitationData(
                        page=current_page,
                        line_start=current_line_num,
                        line_end=current_line_num,
                        bates=current_bates,
                        type="transcript_line"
                    )
                    citation_map[f"line_{idx}"] = citation_data.to_dict()

                    # Add line marker and text
                    enhanced_lines.append(f"[LINE:{current_line_num}]")
                    enhanced_lines.append(f"{current_line_num}  {qa_marker}  {text}")
                    continue

            # Detect patent column markers
            if doc_type in patent_types:
                col_match = re.search(r'col(?:umn)?\.?\s*(\d+)', line, re.IGNORECASE)
                if col_match:
                    column = int(col_match.group(1))
                    enhanced_lines.append(f"[COLUMN:{column}]")

            # Detect paragraph markers
            para_match = re.match(r'^([¶§]\s*\d+)', line)
            if para_match:
                para_num = int(re.search(r'\d+', para_match.group(1)).group(0))
                enhanced_lines.append(f"[PARA:{para_num}]")

                # Create citation entry
                citation_data = CitationData(
                    page=current_page,
                    paragraph_number=para_num,
                    bates=current_bates,
                    type="paragraph"
                )
                citation_map[f"line_{idx}"] = citation_data.to_dict()

            # Add original line
            enhanced_lines.append(line)

        enhanced_content = '\n'.join(enhanced_lines)
        return enhanced_content, citation_map


def main():
    """Test post-processor on a sample markdown file."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python post_processor.py <markdown_file>")
        sys.exit(1)

    processor = PostProcessor()
    result = processor.process(sys.argv[1])

    print(result)


if __name__ == "__main__":
    main()
