"""
Generic document type classifier with self-learning for litigation documents.

Classifies PDFs BEFORE Docling conversion using PyMuPDF-based signals:
1. Filename patterns
2. Content keywords (first 5 pages)
3. Structural patterns (Q/A, margin numbers, columns)
4. Font analysis
5. Learned profiles from previous user corrections

When unsure, can prompt the user interactively and learn from the answer.
"""

import json
import logging
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from citation_types import DocumentType

logger = logging.getLogger(__name__)

PROFILE_DIR = Path.home() / ".lit-pipeline"
PROFILE_PATH = PROFILE_DIR / "type_profiles.json"

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.12
MARGIN_THRESHOLD = 0.04


@dataclass
class ClassificationResult:
    """Result of document classification."""
    doc_type: DocumentType
    confidence: float  # 0.0 - 1.0
    is_text_based: bool  # True if avg chars/page > 100
    signals: Dict[str, Any] = field(default_factory=dict)
    needs_user_input: bool = False


@dataclass
class DocumentFingerprint:
    """Structural fingerprint for profile learning (no case-specific content)."""
    monospace_ratio: float = 0.0
    serif_ratio: float = 0.0
    sans_ratio: float = 0.0
    bold_ratio: float = 0.0
    qa_line_density: float = 0.0
    paragraph_marker_density: float = 0.0
    section_number_density: float = 0.0
    avg_line_length: float = 0.0
    avg_chars_per_page: float = 0.0
    has_margin_line_numbers: bool = False
    has_two_columns: bool = False
    has_email_headers: bool = False
    has_court_caption: bool = False
    has_qa_format: bool = False
    has_perjury_language: bool = False
    has_whereas_clauses: bool = False
    has_patent_claims: bool = False
    has_sec_items: bool = False
    has_academic_structure: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentFingerprint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ProfileStore:
    """Persists learned type profiles to ~/.lit-pipeline/type_profiles.json."""

    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self.profiles: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self.profiles = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.profiles = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def has_profiles(self) -> bool:
        return bool(self.profiles)

    def update(self, doc_type: str, fingerprint: DocumentFingerprint):
        """Update running mean/std for a document type with a new fingerprint."""
        fp_dict = fingerprint.to_dict()
        if doc_type not in self.profiles:
            self.profiles[doc_type] = {
                "count": 1,
                "avg_fingerprint": fp_dict,
                "std_fingerprint": {k: 0.0 for k in fp_dict},
            }
        else:
            prof = self.profiles[doc_type]
            n = prof["count"]
            avg = prof["avg_fingerprint"]
            std = prof["std_fingerprint"]

            new_n = n + 1
            for k, v in fp_dict.items():
                old_avg = avg.get(k, 0.0)
                if isinstance(v, bool):
                    v = float(v)
                if isinstance(old_avg, bool):
                    old_avg = float(old_avg)
                new_avg = old_avg + (v - old_avg) / new_n
                # Welford's online variance
                old_std = std.get(k, 0.0)
                new_std = old_std + (v - old_avg) * (v - new_avg)
                avg[k] = new_avg
                std[k] = new_std

            prof["count"] = new_n

        self.save()

    def score_fingerprint(self, fingerprint: DocumentFingerprint) -> Dict[str, float]:
        """Score a fingerprint against all stored profiles. Returns {type: score}."""
        scores = {}
        fp_dict = fingerprint.to_dict()

        for doc_type, prof in self.profiles.items():
            if prof["count"] < 1:
                continue

            avg = prof["avg_fingerprint"]
            std_raw = prof["std_fingerprint"]
            n = prof["count"]

            distance = 0.0
            num_features = 0

            for k, v in fp_dict.items():
                if k not in avg:
                    continue
                mean = avg[k]
                if isinstance(v, bool):
                    v = float(v)
                if isinstance(mean, bool):
                    mean = float(mean)

                # Compute std from Welford's running sum
                variance = std_raw.get(k, 0.0) / n if n > 1 else 0.01
                std = max(variance ** 0.5, 0.01)  # floor at 0.01

                distance += ((v - mean) / std) ** 2
                num_features += 1

            if num_features > 0:
                # Convert distance to similarity score (0-1)
                avg_dist = (distance / num_features) ** 0.5
                scores[doc_type] = max(0.0, 1.0 / (1.0 + avg_dist))

        return scores


# ── Signal functions ──────────────────────────────────────────────────

FILENAME_PATTERNS: List[Tuple[re.Pattern, DocumentType]] = [
    (re.compile(r"deposition|condensed.?transcript|transcript.?of", re.I), DocumentType.DEPOSITION),
    (re.compile(r"hearing.?transcript", re.I), DocumentType.HEARING_TRANSCRIPT),
    (re.compile(r"expert.?report|rebuttal.?report", re.I), DocumentType.EXPERT_REPORT),
    (re.compile(r"declaration.?of|decl\.?\s+of", re.I), DocumentType.DECLARATION),
    (re.compile(r"motion.?to|mot\.", re.I), DocumentType.MOTION),
    (re.compile(r"\bbrief\b|opposition|reply\b", re.I), DocumentType.BRIEF),
    (re.compile(r"\bpatent\b", re.I), DocumentType.PATENT),
    (re.compile(r"office.?action|file.?wrapper|prosecution", re.I), DocumentType.PATENT_PROSECUTION),
    (re.compile(r"agreement|license|contract|settlement", re.I), DocumentType.AGREEMENT),
    (re.compile(r"10-[KQ]|SEC\b|annual.?report", re.I), DocumentType.SEC_FILING),
    (re.compile(r"\bIEEE\b|standard|specification", re.I), DocumentType.TECHNICAL_STANDARD),
    (re.compile(r"statement.?of|witness.?statement|affidavit", re.I), DocumentType.WITNESS_STATEMENT),
    # Discovery responses: "Resp(onse)? to ROGs/RFPs/RFAs", "Obj. and Resp.", "Supp Resp", etc.
    (re.compile(
        r"resp(?:onses?|s)?\s+(?:and\s+objs?|to)\s+.*(?:rogs?|interrog|rfps?|requests?\s+for\s+production|rfas?|requests?\s+for\s+admiss)"
        r"|obj(?:\.|ections?)?\s+and\s+resp(?:onses?|s)?"
        r"|supp(?:l(?:emental)?)?\s+resp"
        r"|\b(?:rogs?|rfps?|rfas?)\s+no",
        re.I,
    ), DocumentType.DISCOVERY_RESPONSE),
    # Discovery requests: "First/Second/N-th Set of ROGs/RFPs/RFAs", "Notice of Deposition"
    (re.compile(
        r"(?:first|second|third|fourth|fifth|\d(?:st|nd|rd|th)?)\s+set\s+of\s+(?:rogs?|interrog|rfps?|requests?|rfas?)"
        r"|set\s+of\s+(?:interrogatories|requests?\s+for\s+(?:production|admission))",
        re.I,
    ), DocumentType.DISCOVERY_REQUEST),
]

CONTENT_KEYWORDS: Dict[DocumentType, List[str]] = {
    DocumentType.DEPOSITION: [
        "DEPOSITION OF", "TRANSCRIPT OF PROCEEDINGS", "COURT REPORTER",
        "VIDEOGRAPHER", "CCR", "RPR", "CRR", "CSR", "SWORN TESTIMONY",
    ],
    DocumentType.HEARING_TRANSCRIPT: [
        "HEARING", "PROCEEDINGS", "THE COURT", "YOUR HONOR",
        "ARBITRATION", "ORAL ARGUMENT",
    ],
    DocumentType.EXPERT_REPORT: [
        "EXPERT REPORT", "OPINION", "METHODOLOGY", "ANALYSIS",
        "RETAINED BY", "QUALIFICATIONS",
    ],
    DocumentType.DECLARATION: [
        "DECLARATION OF", "UNDER PENALTY OF PERJURY", "DECLARANT",
        "I DECLARE",
    ],
    DocumentType.MOTION: [
        "MOTION TO", "MOVES THIS COURT", "IN SUPPORT OF",
        "MEMORANDUM OF POINTS",
    ],
    DocumentType.BRIEF: [
        "BRIEF IN", "TABLE OF AUTHORITIES", "ARGUMENT",
        "STATEMENT OF ISSUES",
    ],
    DocumentType.PATENT: [
        "UNITED STATES PATENT", "CLAIMS", "ABSTRACT", "INVENTORS",
        "FIELD OF THE INVENTION", "PRIOR ART",
    ],
    DocumentType.PATENT_PROSECUTION: [
        "OFFICE ACTION", "NOTICE OF ALLOWANCE", "INFORMATION DISCLOSURE",
        "CLAIMS REJECTED", "AMENDMENT",
    ],
    DocumentType.AGREEMENT: [
        "WHEREAS", "NOW THEREFORE", "WITNESSETH", "LICENSE AGREEMENT",
        "EFFECTIVE DATE", "HEREBY AGREES",
    ],
    DocumentType.CORRESPONDENCE: [
        "FROM:", "TO:", "SUBJECT:", "DATE:", "DEAR",
    ],
    DocumentType.TECHNICAL_STANDARD: [
        "IEEE", "NORMATIVE REFERENCES", "SCOPE",
        "SHALL", "CONFORMANCE",
    ],
    DocumentType.SCHOLARLY: [
        "ABSTRACT", "REFERENCES", "INTRODUCTION", "CONCLUSION",
        "DOI", "ISSN", "UNIVERSITY",
    ],
    DocumentType.SEC_FILING: [
        "SECURITIES AND EXCHANGE", "10-K", "ANNUAL REPORT",
        "FORM 10-Q", "ITEM 1",
    ],
    DocumentType.WITNESS_STATEMENT: [
        "STATEMENT OF", "AFFIDAVIT", "SWORN", "NOTARIZED",
    ],
    DocumentType.DISCOVERY_RESPONSE: [
        "INTERROGATORY NO.", "RESPONSE TO INTERROGATORY",
        "REQUEST FOR PRODUCTION NO.", "RESPONSE TO REQUEST FOR PRODUCTION",
        "REQUEST FOR ADMISSION NO.", "RESPONSE TO REQUEST FOR ADMISSION",
        "GENERAL OBJECTIONS", "SUPPLEMENTAL RESPONSE",
        "PURSUANT TO RULES 26 AND 33", "PURSUANT TO RULE 33",
        "PURSUANT TO RULE 34", "PURSUANT TO RULE 36",
        "OBJECTIONS AND RESPONSES",
    ],
    DocumentType.DISCOVERY_REQUEST: [
        "INTERROGATORY NO.", "REQUEST FOR PRODUCTION NO.",
        "REQUEST FOR ADMISSION NO.", "PROPOUNDS THE FOLLOWING",
        "FIRST SET OF INTERROGATORIES", "SECOND SET OF INTERROGATORIES",
        "FIRST SET OF REQUESTS", "SECOND SET OF REQUESTS",
        "NOTICE OF DEPOSITION",
    ],
}


def _signal_filename(filename: str) -> Dict[DocumentType, float]:
    """Signal 1: Classify based on filename patterns."""
    scores: Dict[DocumentType, float] = defaultdict(float)
    name_lower = filename.lower()

    # Check for Bates prefix pattern (weak exhibit signal)
    if re.match(r"[a-z]+_[a-z]+_\d{5,}", name_lower):
        scores[DocumentType.EXHIBIT] = 0.15

    for pattern, doc_type in FILENAME_PATTERNS:
        if pattern.search(name_lower):
            scores[doc_type] = max(scores[doc_type], 1.0)

    # Mutual exclusion: filenames like "Resp to 2nd Set of Rogs" trigger
    # both DISCOVERY_RESPONSE and DISCOVERY_REQUEST. The presence of "Resp"
    # always wins — it's the response to that set, not the set itself.
    if scores.get(DocumentType.DISCOVERY_RESPONSE, 0) >= 1.0:
        scores.pop(DocumentType.DISCOVERY_REQUEST, None)

    return dict(scores)


def _signal_content(text: str) -> Dict[DocumentType, float]:
    """Signal 2: Classify based on content keywords from first 5 pages."""
    scores: Dict[DocumentType, float] = defaultdict(float)
    text_upper = text.upper()

    for doc_type, keywords in CONTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_upper)
        if hits > 0:
            scores[doc_type] = min(1.0, hits / max(3, len(keywords) * 0.4))

    return dict(scores)


def _signal_structural(text: str, num_pages: int) -> Tuple[Dict[DocumentType, float], dict]:
    """Signal 3: Classify based on structural patterns."""
    scores: Dict[DocumentType, float] = defaultdict(float)
    debug: Dict[str, Any] = {}

    lines = text.split("\n")
    num_pages = max(num_pages, 1)

    # Q/A line counting
    qa_count = sum(1 for l in lines if re.match(r"^\s*\d{0,2}\s*[QA][.\s:]", l))
    debug["qa_lines"] = qa_count
    if qa_count > 10:
        scores[DocumentType.DEPOSITION] = min(1.0, qa_count / 30)
        scores[DocumentType.HEARING_TRANSCRIPT] = min(0.5, qa_count / 60)

    # Margin line numbers (digits 1-25 at start of line)
    margin_nums = sum(1 for l in lines if re.match(r"^\s{0,4}\d{1,2}\s{2,}", l))
    debug["margin_line_numbers"] = margin_nums
    if margin_nums > 20:
        scores[DocumentType.DEPOSITION] = max(scores.get(DocumentType.DEPOSITION, 0), 0.7)

    # Dense numbered paragraphs
    numbered_paras = sum(1 for l in lines if re.match(r"^\d+\.\s+[A-Z]", l))
    debug["numbered_paragraphs"] = numbered_paras
    if numbered_paras > 5:
        para_density = numbered_paras / num_pages
        scores[DocumentType.EXPERT_REPORT] = min(1.0, para_density / 3)
        scores[DocumentType.DECLARATION] = min(0.6, para_density / 5)

    # Section hierarchy (N.N.N patterns)
    section_nums = sum(1 for l in lines if re.match(r"^\d+\.\d+\.?\d*\.?\s+", l))
    debug["section_numbers"] = section_nums
    if section_nums > 5:
        scores[DocumentType.TECHNICAL_STANDARD] = min(1.0, section_nums / 15)
        scores[DocumentType.AGREEMENT] = min(0.5, section_nums / 20)

    # Email header block
    email_headers = sum(1 for l in lines[:30] if re.match(r"^(From|To|Subject|Date|Cc|Bcc)\s*:", l))
    debug["email_headers"] = email_headers
    if email_headers >= 3:
        scores[DocumentType.CORRESPONDENCE] = 1.0

    # SEC "ITEM N" pattern
    item_count = sum(1 for l in lines if re.match(r"^\s*ITEM\s+\d+", l, re.I))
    debug["sec_items"] = item_count
    if item_count >= 2:
        scores[DocumentType.SEC_FILING] = min(1.0, item_count / 4)

    # Paragraph markers (¶ or §)
    para_markers = sum(1 for l in lines if re.search(r"[¶§]\s*\d+", l))
    debug["para_markers"] = para_markers
    if para_markers > 3:
        scores[DocumentType.EXPERT_REPORT] = max(scores.get(DocumentType.EXPERT_REPORT, 0), 0.5)

    # Discovery request boundary headers ("INTERROGATORY NO. 1", "REQUEST FOR
    # PRODUCTION NO. 12", "REQUEST FOR ADMISSION NO. 5"). Strong + specific —
    # these patterns essentially never occur outside discovery docs.
    discovery_boundaries = sum(
        1 for l in lines
        if re.match(
            r"^\s*(?:INTERROGATORY|REQUEST\s+FOR\s+(?:PRODUCTION|ADMISSION))\s+NO\.\s*\d+",
            l, re.I,
        )
    )
    debug["discovery_boundaries"] = discovery_boundaries
    if discovery_boundaries >= 1:
        # Each numbered request is worth roughly 1/3 of a confidence point;
        # a doc with 3+ requests is strongly classified.
        score = min(1.0, discovery_boundaries / 3.0)
        # If the doc also contains "RESPONSE TO" markers, it's a response;
        # otherwise it's a request. Looking only at first 5 pages so this
        # is a heuristic, not a guarantee — content keywords give the final say.
        text_upper = text.upper()
        if (
            "RESPONSE TO INTERROGATORY" in text_upper
            or "RESPONSE TO REQUEST" in text_upper
            or "OBJECTIONS AND RESPONSES" in text_upper
            or "GENERAL OBJECTIONS" in text_upper
            or "SUPPLEMENTAL RESPONSE" in text_upper
        ):
            scores[DocumentType.DISCOVERY_RESPONSE] = score
        else:
            scores[DocumentType.DISCOVERY_REQUEST] = score

    return dict(scores), debug


def _signal_font(doc: fitz.Document, max_pages: int = 5) -> Tuple[Dict[DocumentType, float], dict]:
    """Signal 4: Classify based on font analysis."""
    scores: Dict[DocumentType, float] = defaultdict(float)

    monospace_chars = 0
    serif_chars = 0
    sans_chars = 0
    bold_chars = 0
    total_chars = 0

    pages_to_check = min(len(doc), max_pages)
    for page_idx in range(pages_to_check):
        page = doc[page_idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # text block
                continue
            for line_data in block.get("lines", []):
                for span in line_data.get("spans", []):
                    font_name = span.get("font", "").lower()
                    char_count = len(span.get("text", ""))
                    total_chars += char_count

                    if any(m in font_name for m in ("courier", "mono", "consolas", "menlo")):
                        monospace_chars += char_count
                    elif any(s in font_name for s in ("times", "garamond", "georgia", "palatino", "serif")):
                        serif_chars += char_count
                    elif any(s in font_name for s in ("arial", "helvetica", "calibri", "sans")):
                        sans_chars += char_count

                    if "bold" in font_name or span.get("flags", 0) & 2 ** 4:
                        bold_chars += char_count

    debug = {}
    if total_chars > 0:
        mono_ratio = monospace_chars / total_chars
        serif_ratio = serif_chars / total_chars
        bold_ratio = bold_chars / total_chars
        debug = {
            "monospace_ratio": round(mono_ratio, 2),
            "serif_ratio": round(serif_ratio, 2),
            "sans_ratio": round(sans_chars / total_chars, 2),
            "bold_ratio": round(bold_ratio, 2),
        }

        if mono_ratio > 0.4:
            scores[DocumentType.DEPOSITION] = min(1.0, mono_ratio)
            scores[DocumentType.HEARING_TRANSCRIPT] = min(0.5, mono_ratio * 0.6)

    return dict(scores), debug


def _extract_fingerprint(
    doc: fitz.Document, text: str, num_pages: int
) -> DocumentFingerprint:
    """Extract a structural fingerprint for profile learning."""
    lines = text.split("\n")
    num_pages = max(num_pages, 1)

    # Font features
    mono_chars = serif_chars = sans_chars = bold_chars = total_chars = 0
    pages_to_check = min(len(doc), 5)
    for page_idx in range(pages_to_check):
        page = doc[page_idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line_data in block.get("lines", []):
                for span in line_data.get("spans", []):
                    font_name = span.get("font", "").lower()
                    cc = len(span.get("text", ""))
                    total_chars += cc
                    if any(m in font_name for m in ("courier", "mono", "consolas", "menlo")):
                        mono_chars += cc
                    elif any(s in font_name for s in ("times", "garamond", "georgia", "palatino", "serif")):
                        serif_chars += cc
                    elif any(s in font_name for s in ("arial", "helvetica", "calibri", "sans")):
                        sans_chars += cc
                    if "bold" in font_name or span.get("flags", 0) & 2 ** 4:
                        bold_chars += cc

    tc = max(total_chars, 1)

    # Content features
    qa_lines = sum(1 for l in lines if re.match(r"^\s*\d{0,2}\s*[QA][.\s:]", l))
    para_markers = sum(1 for l in lines if re.search(r"[¶§]\s*\d+", l))
    section_nums = sum(1 for l in lines if re.match(r"^\d+\.\d+\.?\d*\.?\s+", l))
    line_lengths = [len(l) for l in lines if l.strip()]
    avg_line_len = statistics.mean(line_lengths) if line_lengths else 0

    # Two-column detection via block x-positions
    has_two_cols = False
    if len(doc) > 0:
        x_positions = []
        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox", [0, 0, 0, 0])
            center_x = (bbox[0] + bbox[2]) / 2
            x_positions.append(center_x)
        if len(x_positions) >= 6:
            x_sorted = sorted(x_positions)
            mid = len(x_sorted) // 2
            left_median = statistics.median(x_sorted[:mid])
            right_median = statistics.median(x_sorted[mid:])
            page_width = doc[0].rect.width
            if right_median - left_median > page_width * 0.25:
                has_two_cols = True

    text_upper = text.upper()

    return DocumentFingerprint(
        monospace_ratio=mono_chars / tc,
        serif_ratio=serif_chars / tc,
        sans_ratio=sans_chars / tc,
        bold_ratio=bold_chars / tc,
        qa_line_density=qa_lines / num_pages,
        paragraph_marker_density=para_markers / num_pages,
        section_number_density=section_nums / num_pages,
        avg_line_length=avg_line_len,
        avg_chars_per_page=len(text) / num_pages,
        has_margin_line_numbers=sum(1 for l in lines if re.match(r"^\s{0,4}\d{1,2}\s{2,}", l)) > 20,
        has_two_columns=has_two_cols,
        has_email_headers=sum(1 for l in lines[:30] if re.match(r"^(From|To|Subject|Date)\s*:", l)) >= 3,
        has_court_caption=bool(re.search(r"IN THE .{5,40} COURT", text_upper)),
        has_qa_format=qa_lines > 10,
        has_perjury_language="UNDER PENALTY OF PERJURY" in text_upper,
        has_whereas_clauses="WHEREAS" in text_upper and "NOW THEREFORE" in text_upper,
        has_patent_claims="UNITED STATES PATENT" in text_upper or ("CLAIMS" in text_upper and "ABSTRACT" in text_upper),
        has_sec_items=bool(re.search(r"ITEM\s+\d+", text_upper)),
        has_academic_structure="ABSTRACT" in text_upper and "REFERENCES" in text_upper,
    )


# ── Main classifier ──────────────────────────────────────────────────

def classify_document(
    pdf_path: str,
    profile_store: Optional[ProfileStore] = None,
) -> ClassificationResult:
    """
    Classify a PDF document by type using PyMuPDF-based signals.

    Args:
        pdf_path: Path to PDF file.
        profile_store: Optional ProfileStore for learned profiles.

    Returns:
        ClassificationResult with doc_type, confidence, and signals.
    """
    pdf_path = str(pdf_path)
    filename = Path(pdf_path).name

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning("Cannot open %s: %s", pdf_path, e)
        return ClassificationResult(
            doc_type=DocumentType.UNKNOWN,
            confidence=0.0,
            is_text_based=False,
            signals={"error": str(e)},
            needs_user_input=True,
        )

    num_pages = len(doc)

    # Extract text from first 5 pages
    max_pages = min(num_pages, 5)
    text_parts = []
    total_chars = 0
    for i in range(max_pages):
        page_text = doc[i].get_text()
        text_parts.append(page_text)
        total_chars += len(page_text)

    text = "\n".join(text_parts)
    is_text_based = (total_chars / max(max_pages, 1)) > 100

    # Gather signals
    s1 = _signal_filename(filename)
    s2 = _signal_content(text)
    s3_scores, s3_debug = _signal_structural(text, max_pages)
    s4_scores, s4_debug = _signal_font(doc, max_pages)

    # Learned profile signal
    s5_scores: Dict[str, float] = {}
    has_profiles = False
    if profile_store and profile_store.has_profiles():
        has_profiles = True
        fingerprint = _extract_fingerprint(doc, text, max_pages)
        s5_scores = profile_store.score_fingerprint(fingerprint)

    doc.close()

    # Aggregate scores with weights
    if has_profiles:
        weights = (0.15, 0.25, 0.25, 0.10, 0.25)
    else:
        weights = (0.20, 0.30, 0.30, 0.15, 0.0)
    # Renormalize if no profile
    w_total = sum(weights)

    all_types = set()
    for s in [s1, s2, s3_scores, s4_scores]:
        all_types.update(s.keys())
    for t_str in s5_scores:
        try:
            all_types.add(DocumentType(t_str))
        except ValueError:
            pass

    final_scores: Dict[DocumentType, float] = {}
    for dt in all_types:
        score = 0.0
        score += weights[0] * s1.get(dt, 0.0)
        score += weights[1] * s2.get(dt, 0.0)
        score += weights[2] * s3_scores.get(dt, 0.0)
        score += weights[3] * s4_scores.get(dt, 0.0)
        if has_profiles:
            score += weights[4] * s5_scores.get(dt.value, 0.0)
        final_scores[dt] = score / w_total

    # Determine winner
    if not final_scores:
        return ClassificationResult(
            doc_type=DocumentType.UNKNOWN,
            confidence=0.0,
            is_text_based=is_text_based,
            signals={"filename": s1, "content": s2, "structural": s3_debug, "font": s4_debug},
            needs_user_input=True,
        )

    sorted_types = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    winner, winner_score = sorted_types[0]
    runner_up_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0
    margin = winner_score - runner_up_score

    # Real depositions trigger filename and/or content signals ("Deposition of
    # X", "TRANSCRIPT OF PROCEEDINGS", etc.). Demote DEPOSITION winners that
    # rely solely on the structural monospace signal, since that fires for
    # patent apps and scholarly articles too — and DEPOSITION uniquely
    # triggers the PyMuPDF fast path and Q&A-aware chunking, which collapse
    # the whole document into one chunk.
    if winner == DocumentType.DEPOSITION and (
        s1.get(DocumentType.DEPOSITION, 0) == 0
        and s2.get(DocumentType.DEPOSITION, 0) == 0
    ):
        winner = DocumentType.UNKNOWN
    elif winner_score == 0.0:
        winner = DocumentType.UNKNOWN

    needs_input = winner_score < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD

    signals = {
        "filename": s1,
        "content": {k.value: v for k, v in s2.items()} if s2 else {},
        "structural": s3_debug,
        "font": s4_debug,
        "final_scores": {k.value: round(v, 3) for k, v in sorted_types[:5]},
    }

    return ClassificationResult(
        doc_type=winner,
        confidence=round(winner_score, 3),
        is_text_based=is_text_based,
        signals=signals,
        needs_user_input=needs_input,
    )


# ── Interactive prompting ─────────────────────────────────────────────

ALL_DOC_TYPES = [
    DocumentType.DEPOSITION,
    DocumentType.HEARING_TRANSCRIPT,
    DocumentType.EXPERT_REPORT,
    DocumentType.DECLARATION,
    DocumentType.MOTION,
    DocumentType.BRIEF,
    DocumentType.PATENT,
    DocumentType.PATENT_PROSECUTION,
    DocumentType.AGREEMENT,
    DocumentType.CORRESPONDENCE,
    DocumentType.TECHNICAL_STANDARD,
    DocumentType.SCHOLARLY,
    DocumentType.SEC_FILING,
    DocumentType.WITNESS_STATEMENT,
    DocumentType.DISCOVERY_REQUEST,
    DocumentType.DISCOVERY_RESPONSE,
    DocumentType.EXHIBIT,
]


def _get_first_page_text(pdf_path: str) -> str:
    """Extract text from the first page of a PDF for display."""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return "(empty document)"
        text = doc[0].get_text().strip()
        doc.close()
        if not text:
            return "(no extractable text — scanned/image-based document)"
        # Truncate to ~40 lines for readability
        lines = text.split("\n")
        if len(lines) > 40:
            lines = lines[:40] + ["  ..."]
        return "\n".join(lines)
    except Exception as e:
        return f"(error reading page 1: {e})"


def prompt_user_for_type(
    pdf_path: str,
    result: ClassificationResult,
) -> DocumentType:
    """Interactively prompt user to confirm or correct document type."""
    print(f"\n{'─'*60}")
    print(f"  File: {Path(pdf_path).resolve()}")
    print(f"  Best guess: {result.doc_type.value} (confidence: {result.confidence:.2f})")

    top_scores = result.signals.get("final_scores", {})
    if top_scores:
        top_items = list(top_scores.items())[:3]
        print(f"  Top scores: {', '.join(f'{t}={s:.2f}' for t, s in top_items)}")

    # Show first page text
    first_page = _get_first_page_text(pdf_path)
    print(f"\n  ── Page 1 preview ──")
    for line in first_page.split("\n"):
        print(f"  │ {line}")
    print(f"  ── end preview ──")

    print("\n  What type of document is this?")
    for i, dt in enumerate(ALL_DOC_TYPES, 1):
        label = f"[{i:2d}] {dt.value}"
        print(f"  {label:30s}", end="")
        if i % 3 == 0:
            print()
    skip_num = len(ALL_DOC_TYPES) + 1
    print(f"\n  [{skip_num}] skip (keep as unknown)")

    try:
        choice = input("  > ").strip()
        idx = int(choice)
        if 1 <= idx <= len(ALL_DOC_TYPES):
            return ALL_DOC_TYPES[idx - 1]
    except (ValueError, EOFError):
        pass

    return DocumentType.UNKNOWN


def classify_directory(
    input_dir: str,
    interactive: bool = True,
    profile_store: Optional[ProfileStore] = None,
) -> Dict[str, ClassificationResult]:
    """
    Classify all PDFs in a directory.

    Args:
        input_dir: Directory containing PDF files.
        interactive: If True (default), prompt user for low-confidence classifications.
        profile_store: Optional ProfileStore for learned profiles.

    Returns:
        Dict mapping normalized stem to ClassificationResult.
    """
    from run_pipeline import normalize_stem

    input_path = Path(input_dir)
    pdfs = sorted(input_path.rglob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF files found in %s", input_dir)
        return {}

    if profile_store is None:
        profile_store = ProfileStore()

    classifications: Dict[str, ClassificationResult] = {}
    needs_input: List[Tuple[str, str, ClassificationResult]] = []

    for pdf in pdfs:
        normalized = normalize_stem(pdf.stem)
        result = classify_document(str(pdf), profile_store)
        classifications[normalized] = result

        logger.info(
            "Classified %s -> %s (confidence: %.2f%s)",
            pdf.name, result.doc_type.value, result.confidence,
            " NEEDS INPUT" if result.needs_user_input else "",
        )

        if result.needs_user_input:
            needs_input.append((str(pdf), normalized, result))

    # Interactive prompting for low-confidence results
    if interactive and needs_input:
        print(f"\n{'='*60}")
        print(f"  {len(needs_input)} document(s) need classification:")
        print(f"{'='*60}")

        for pdf_path, normalized, result in needs_input:
            corrected_type = prompt_user_for_type(pdf_path, result)
            if corrected_type != DocumentType.UNKNOWN:
                result.doc_type = corrected_type
                result.needs_user_input = False

                # Learn from user correction
                try:
                    doc = fitz.open(pdf_path)
                    max_pages = min(len(doc), 5)
                    text_parts = [doc[i].get_text() for i in range(max_pages)]
                    text = "\n".join(text_parts)
                    fingerprint = _extract_fingerprint(doc, text, max_pages)
                    doc.close()
                    profile_store.update(corrected_type.value, fingerprint)
                    logger.info("Learned profile for %s from %s", corrected_type.value, Path(pdf_path).name)
                except Exception as e:
                    logger.warning("Could not learn from %s: %s", pdf_path, e)

    return classifications


# ── Condensed transcript detection ───────────────────────────────────

_CONDENSED_FILENAME_RE = re.compile(r"condensed", re.IGNORECASE)
_CONDENSED_HEADER_RE = re.compile(
    r"\d+\s*\(\s*\d+\s+to\s+\d+\s*\)",  # e.g. "1 (1 to 4)"
)


def is_condensed_transcript(pdf_path: str) -> bool:
    """Detect condensed transcripts (multiple transcript pages per physical page).

    Checks filename for "condensed" and/or first-page content for the
    characteristic "N (N to N)" header pattern.  These documents produce
    garbage when extracted because columns from different pages get merged.

    Returns:
        True if the PDF appears to be a condensed transcript.
    """
    path = Path(pdf_path)

    # Fast check: filename
    if _CONDENSED_FILENAME_RE.search(path.stem):
        return True

    # Content check: look for "N (N to N)" pattern in first 3 pages
    try:
        doc = fitz.open(pdf_path)
        pages_to_check = min(3, len(doc))
        for i in range(pages_to_check):
            text = doc[i].get_text("text")
            if _CONDENSED_HEADER_RE.search(text):
                doc.close()
                return True
        doc.close()
    except Exception:
        pass

    return False
