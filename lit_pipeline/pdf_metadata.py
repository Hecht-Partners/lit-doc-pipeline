"""
PDF metadata extraction using PyMuPDF.

Extracts author, creation date, modified date, and other metadata from PDF files.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_pdf_metadata(pdf_path: Path) -> Dict[str, Optional[str]]:
    """
    Extract metadata from PDF file using PyMuPDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict with keys: author, creation_date, modified_date, producer, title, subject
        All values are strings or None if not available
    """
    try:
        import pymupdf as fitz
    except ImportError:
        logger.warning("PyMuPDF not available, skipping metadata extraction")
        return {
            "author": None,
            "creation_date": None,
            "modified_date": None,
            "producer": None,
            "title": None,
            "subject": None,
        }

    try:
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata

        # Extract and clean metadata fields
        result = {
            "author": _clean_metadata_string(metadata.get("author")),
            "creation_date": _parse_pdf_date(metadata.get("creationDate")),
            "modified_date": _parse_pdf_date(metadata.get("modDate")),
            "producer": _clean_metadata_string(metadata.get("producer")),
            "title": _clean_metadata_string(metadata.get("title")),
            "subject": _clean_metadata_string(metadata.get("subject")),
        }

        doc.close()
        return result

    except Exception as e:
        logger.warning(f"Failed to extract metadata from {pdf_path.name}: {e}")
        return {
            "author": None,
            "creation_date": None,
            "modified_date": None,
            "producer": None,
            "title": None,
            "subject": None,
        }


def _clean_metadata_string(value: Optional[str]) -> Optional[str]:
    """
    Clean metadata string (remove empty strings, whitespace).

    Args:
        value: Raw metadata value

    Returns:
        Cleaned string or None if empty
    """
    if not value:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    return cleaned


def _parse_pdf_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parse PDF date format to ISO 8601.

    PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm
    Example: D:20240115113045-05'00

    Args:
        date_str: PDF date string

    Returns:
        ISO 8601 date string (YYYY-MM-DDTHH:mm:ss) or None if invalid
    """
    if not date_str:
        return None

    try:
        # Remove 'D:' prefix if present
        if date_str.startswith("D:"):
            date_str = date_str[2:]

        # Extract components (at least YYYYMMDD required)
        if len(date_str) < 8:
            return None

        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]

        # Optional time components
        hour = date_str[8:10] if len(date_str) >= 10 else "00"
        minute = date_str[10:12] if len(date_str) >= 12 else "00"
        second = date_str[12:14] if len(date_str) >= 14 else "00"

        # Construct ISO 8601 date
        iso_date = f"{year}-{month}-{day}T{hour}:{minute}:{second}"

        # Validate by parsing
        datetime.fromisoformat(iso_date)

        return iso_date

    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse PDF date '{date_str}': {e}")
        return None


def get_page_count(pdf_path: Path) -> Optional[int]:
    """
    Get page count from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages, or None if error
    """
    try:
        import pymupdf as fitz

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
        return page_count

    except Exception as e:
        logger.warning(f"Failed to get page count from {pdf_path.name}: {e}")
        return None
