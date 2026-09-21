"""
syllabus_parser.py
-------------------
Parses a college curriculum/syllabus PDF (CBCS-style, e.g. Anna University /
Sairam Engineering College format) into a structured list of:

    Subject -> Module/Unit -> Topics -> Sub-topics

and the associated Text Book / Reference Book citations for that subject.

This is regex-based (no LLM needed for this step) because syllabus PDFs of
this kind follow a very predictable layout:

    <CODE> <SUBJECT NAME> ... L T P C
    (optional "Common to ..." line)
    <L> <T> <P> <C>
    OBJECTIVES
    ...
    UNIT I <UNIT TITLE> <hours>
    <topic paragraph, semicolons/commas separated, may contain CO tags>
    UNIT II <UNIT TITLE> <hours>
    ...
    TOTAL: NN PERIODS
    TEXT BOOKS
    1. Author, "Title", edition, Publisher, year.
    ...
    REFERENCE BOOKS
    1. ...
    COURSE OUTCOMES
    ...

Works on any PDF with a similar structure -- tweak the regexes below if your
college's format differs slightly.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class BookRef:
    raw: str
    author: str = ""
    title: str = ""
    edition: str = ""
    publisher: str = ""
    year: str = ""
    book_type: str = "textbook"  # "textbook" or "reference"


@dataclass
class TopicItem:
    topic: str
    subtopics: List[str] = field(default_factory=list)


@dataclass
class Module:
    module_no: int
    module_title: str
    hours: Optional[str]
    raw_text: str
    topics: List[TopicItem] = field(default_factory=list)  # topic -> [subtopics]


@dataclass
class Subject:
    subject_code: str
    subject_name: str
    semester: Optional[str]
    modules: List[Module] = field(default_factory=list)
    text_books: List[BookRef] = field(default_factory=list)
    reference_books: List[BookRef] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# PDF text extraction
# --------------------------------------------------------------------------- #

def get_pdf_text_per_page(pdf_path: str) -> List[str]:
    """Returns a list of page texts. Uses PyMuPDF if available (fast/accurate),
    else falls back to pdfplumber."""
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(pdf_path)
        return [page.get_text() for page in doc]
    except ImportError:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return [p.extract_text() or "" for p in pdf.pages]


# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #

SEMESTER_HEADING_RE = re.compile(r'SEMESTER\s*[\u2013\u2014\-:]*\s*([IVX]+)\b')
MAX_SEMESTER_TABLE_CHARS = 3000  # a real per-semester course table is short;
                                  # anything longer is a different kind of
                                  # listing (e.g. a cumulative summary table)
                                  # that would corrupt the mapping if trusted.

# e.g. "GE4105 PROBLEM SOLVING AND PYTHON PROGRAMMING L T P C"
SUBJECT_HEADER_RE = re.compile(
    r'(?P<code>[A-Z]{2,4}\d{3,5})\s+(?P<name>[A-Z][A-Z0-9 &/(),.\-]{4,100}?)\s+L\s*T\s*P\s*C',
)

# e.g. "UNIT I ALGORITHMIC PROBLEM SOLVING 9"  or "UNIT III ... 7+12"
UNIT_RE = re.compile(
    r'UNIT\s+([IVX]+)\s+([A-Z][A-Z0-9 &/(),.\-\']{3,100}?)\s+(\d+(?:\+\d+)?)\s*\n',
)

TEXTBOOKS_HDR_RE = re.compile(r'^\s*TEXT\s*BOOKS?\s*$', re.MULTILINE)
REFBOOKS_HDR_RE = re.compile(r'^\s*REFERENCE\s*BOOKS?\s*$', re.MULTILINE)
COURSE_OUTCOMES_HDR_RE = re.compile(r'^\s*COURSE\s+OUTCOMES\s*$', re.MULTILINE)
MAPPING_HDR_RE = re.compile(r'^\s*MAPPING\s+OF\s+COs', re.MULTILINE)

# Numbered citation entries: "1. Author..., "Title", Publisher, Year."
CITATION_ITEM_RE = re.compile(r'\n\s*(\d{1,2})\.\s+', re.MULTILINE)

ROMAN_MAP = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8}


def build_semester_map(full_text: str) -> dict:
    """Maps subject_code -> semester roman numeral by scanning each
    "SEMESTER <roman>" heading's course table (not just the nearest
    preceding heading anywhere in the document).

    Why this matters: syllabus PDFs often print "SEMESTER I" with a plain
    space for some semesters but "SEMESTER\u2013I" with an en-dash for
    others (a Word-to-PDF export inconsistency), AND they frequently
    contain a second, much later "cumulative" listing of every course
    across all 8 semesters under a final heading. A naive "closest heading
    before this position, anywhere in the document" lookup silently
    misattributes early-semester subjects to whatever heading happens to
    survive that inconsistency (in practice, subjects were getting
    stamped "Sem VIII" incorrectly). Scanning each heading's own table span
    for course codes, and keeping only the FIRST (i.e. correct, per-semester
    table) assignment per code, fixes this.
    """
    matches = list(SEMESTER_HEADING_RE.finditer(full_text))
    code_re = re.compile(r'\b([A-Z]{2,4}\d{3,5})\b')
    mapping: dict = {}
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end]
        if len(chunk) > MAX_SEMESTER_TABLE_CHARS:
            continue  # not a real per-semester table -- skip to avoid corrupting the map
        for code in code_re.findall(chunk):
            mapping.setdefault(code, m.group(1))
    return mapping


# --------------------------------------------------------------------------- #
# Citation parsing (best-effort -- publisher catalogs vary a lot in format)
# --------------------------------------------------------------------------- #

YEAR_RE = re.compile(r'(19|20)\d{2}')
EDITION_RE = re.compile(r'(\d+(?:st|nd|rd|th)\s*Ed(?:ition|\.)?)', re.IGNORECASE)


def _fix_missing_spaces(text: str) -> str:
    """Some source PDFs strip spaces when text is pasted in from another
    tool (e.g. "MichaelT.Goodrich,RobertoTamassia" instead of "Michael T.
    Goodrich, Roberto Tamassia"). This heuristically reinserts spaces at
    lowercase->Uppercase and letter->digit boundaries. It won't recover
    spaces lost between two lowercase words (e.g. "Algorithmsin" ->
    "Algorithms in" needs a dictionary to fix), but it recovers the common
    author-name/word-boundary case well."""
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', text)
    text = re.sub(r'(?<=[0-9])(?=[A-Za-z])', ' ', text)
    text = re.sub(r'(?<=\.)(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_citation(raw: str, book_type: str) -> BookRef:
    raw_clean = re.sub(r'\s+', ' ', raw).strip()
    raw_clean = _fix_missing_spaces(raw_clean)

    year_m = YEAR_RE.search(raw_clean)
    year = year_m.group(0) if year_m else ""

    edition_m = EDITION_RE.search(raw_clean)
    edition = edition_m.group(1) if edition_m else ""

    # Title heuristic: text within the first pair of quotes (curly or straight)
    title_m = re.search(r'[\u201c"\'\u2018]([^\u201d"\'\u2019]{4,150})[\u201d"\'\u2019]', raw_clean)
    title = title_m.group(1).strip() if title_m else ""

    # Author heuristic: text before the title/quote, else before first comma
    if title_m:
        author = raw_clean[:title_m.start()].strip(' ,.\u2013-')
    else:
        author = raw_clean.split(',')[0].strip()

    # Publisher heuristic: segment right after title/edition, before year,
    # excluding pure place names is hard -- keep it as the trailing chunk
    # minus the year.
    tail = raw_clean
    if title_m:
        tail = raw_clean[title_m.end():]
    tail = re.sub(EDITION_RE, '', tail)
    if year_m:
        tail = tail[:tail.find(year_m.group(0))]
    publisher = tail.strip(' ,.\u2013-')

    return BookRef(
        raw=raw_clean,
        author=author,
        title=title,
        edition=edition,
        publisher=publisher,
        year=year,
        book_type=book_type,
    )


def split_citations(block_text: str, book_type: str) -> List[BookRef]:
    """Split a TEXT BOOKS / REFERENCE BOOKS block into individual citations."""
    if not block_text.strip():
        return []
    parts = CITATION_ITEM_RE.split('\n' + block_text)
    # parts alternates [prefix, num, entry, num, entry, ...]; drop numbers
    entries = [p for p in parts[2::2]] if len(parts) > 2 else [block_text]
    return [parse_citation(e, book_type) for e in entries if e.strip()]


# --------------------------------------------------------------------------- #
# Topic / sub-topic splitting
# --------------------------------------------------------------------------- #

# Language/communication-skill courses (Communicative English, Professional
# English, etc.) structure every unit around a fixed set of skill sections
# instead of a flat list of concepts. When a unit clearly uses these (we
# require at least 3 matches before switching modes), we split on the
# section keywords themselves rather than on generic punctuation -- this
# is what keeps "Reading" from becoming an orphan topic while "Writing",
# "Listening", "Speaking" etc. get merged/dropped inconsistently.
SECTION_KEYWORDS = [
    "Language development", "Language Development",
    "Vocabulary development", "Vocabulary Development",
    "Reading", "Writing", "Listening", "Speaking", "Grammar", "Pronunciation",
]
SECTION_KEYWORD_RE = re.compile(
    r'(?<![A-Za-z])(' + '|'.join(re.escape(k) for k in SECTION_KEYWORDS) + r')(?![A-Za-z])'
)


def _normalize_body(unit_body: str) -> str:
    """Shared cleanup: strip inline CO tags / trailing TOTAL line, collapse
    whitespace, recover PDF-glued-word spacing, and normalize separator
    hyphens (word- Word) to en-dash without touching compound-word hyphens
    (no space) or word-wrap hyphens (space + lowercase)."""
    text = re.sub(r'\bCO\d+\b', '', unit_body)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'TOTAL\s*:\s*\d+\s*PERIODS.*$', '', text, flags=re.IGNORECASE)
    text = _fix_missing_spaces(text)
    text = re.sub(r'(?<=[a-z])-\s+(?=[A-Z])', '\u2013', text)
    return text


def _split_flat(text: str) -> List[str]:
    """Split a text span into a flat list of phrases, treating ';', en-dash,
    and em-dash as the primary separators, and also splitting long
    (>40 char) comma-bearing segments on commas. Colons are NOT treated as
    a separator here -- callers that want "label: item, item" behaviour
    should split on the first colon themselves before calling this."""
    chunks = [c.strip(' .') for c in re.split(r'[\u2013\u2014;]', text) if c.strip(' .')]
    phrases: List[str] = []
    for chunk in chunks:
        if ',' in chunk and len(chunk) > 40:
            phrases.extend(s.strip(' .-') for s in chunk.split(','))
        else:
            phrases.append(chunk.strip(' .-'))
    return [p for p in phrases if len(p) >= 3]


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = re.sub(r'\s+', ' ', it).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _merge_orphans(items: List[TopicItem], min_len: int = 15) -> List[TopicItem]:
    """Fold very short, sub-topic-less topics (leftover fragments like
    "voices" or "What" from ambiguous dash punctuation) into the previous
    topic's sub-topic list instead of leaving them as standalone noise
    topics. The first item, if short, is left alone since there's nothing
    to merge it into."""
    merged: List[TopicItem] = []
    for item in items:
        if merged and not item.subtopics and len(item.topic) < min_len:
            merged[-1].subtopics.append(item.topic)
        else:
            merged.append(item)
    return merged


def _split_topics_by_section(text: str) -> List[TopicItem]:
    """Language-course mode: one Topic per skill-section keyword
    (Reading/Writing/Listening/Speaking/Language development/Vocabulary
    development/...), with everything up to the next section keyword
    becoming that section's sub-topics."""
    matches = list(SECTION_KEYWORD_RE.finditer(text))
    items: List[TopicItem] = []
    for idx, m in enumerate(matches):
        label = re.sub(r'\s+', ' ', m.group(1)).strip().title()
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip(' :-\u2013')
        if not body:
            continue  # duplicate/empty section label in the source text -- nothing to attach
        subtopics = _dedup_preserve_order(_split_flat(body))
        items.append(TopicItem(topic=label, subtopics=subtopics))
    return items


def _split_topics_general(text: str) -> List[TopicItem]:
    """Technical-course mode: split on ';'/en-dash/em-dash for topic
    boundaries. A colon inside a chunk introduces a sub-list ("Torsion
    pendulum: theory and experiment" -> topic "Torsion pendulum",
    sub-topics ["theory", "experiment"]) rather than being treated as
    another topic boundary, which previously fragmented single concepts."""
    chunks = [c.strip(' .') for c in re.split(r'[\u2013\u2014;]', text) if c.strip(' .')]
    items: List[TopicItem] = []
    for chunk in chunks:
        if ':' in chunk:
            label, _, rest = chunk.partition(':')
            label = label.strip(' .-')
            subs = _dedup_preserve_order([s.strip(' .-') for s in rest.split(',') if len(s.strip(' .-')) >= 3])
            if label:
                items.append(TopicItem(topic=label, subtopics=subs))
            continue

        subs = [s.strip(' .-') for s in chunk.split(',') if len(s.strip(' .-')) >= 3]
        if not subs:
            continue
        if len(subs) == 1:
            items.append(TopicItem(topic=subs[0], subtopics=[]))
        elif len(chunk) <= 90:
            items.append(TopicItem(topic=chunk, subtopics=_dedup_preserve_order(subs)))
        else:
            items.append(TopicItem(topic=subs[0], subtopics=_dedup_preserve_order(subs[1:])))
    return items


def split_topics(unit_body: str) -> List[TopicItem]:
    """Split a unit's descriptive paragraph into a Topic -> [Sub-topics]
    hierarchy, preserving every item in the source text.

    Real syllabus PDFs (this one included) are inconsistent about
    delimiters: some use commas/semicolons, many use en-dashes ("\u2013")
    as the primary list separator (a Word-to-PDF conversion quirk), plain
    ASCII hyphens show up BOTH as separators ("Encapsulation- Data") AND
    inside compound words ("array-based", "non-uniform") that must NOT be
    split apart, and language-skill courses structure content around fixed
    section labels (Reading/Writing/Listening/...) rather than a flat list
    of concepts at all. This function detects which style a unit uses and
    applies the matching splitter, then removes exact duplicate
    topic/sub-topic fragments and folds short orphaned fragments into their
    neighbouring topic.
    """
    text = _normalize_body(unit_body)
    if not text:
        return []

    section_hits = len(SECTION_KEYWORD_RE.findall(text))
    if section_hits >= 3:
        items = _split_topics_by_section(text)
    else:
        items = _split_topics_general(text)

    items = [it for it in items if it.topic]
    items = _merge_orphans(items)

    # final module-wide dedup across topic labels (keep first occurrence)
    seen = set()
    deduped: List[TopicItem] = []
    for it in items:
        key = re.sub(r'\s+', ' ', it.topic).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    if not deduped and text:
        deduped = [TopicItem(topic=text, subtopics=[])]
    return deduped


# --------------------------------------------------------------------------- #
# Main parse routine
# --------------------------------------------------------------------------- #

def parse_syllabus(pdf_path: str) -> List[Subject]:
    pages = get_pdf_text_per_page(pdf_path)
    full_text = "\n".join(pages)

    # Track semester per subject_code via each SEMESTER heading's own table
    # span (see build_semester_map docstring for why this beats a naive
    # nearest-preceding-heading lookup).
    header_matches = list(SUBJECT_HEADER_RE.finditer(full_text))
    semester_map = build_semester_map(full_text)

    subjects: List[Subject] = []

    for idx, hm in enumerate(header_matches):
        start = hm.end()
        end = header_matches[idx + 1].start() if idx + 1 < len(header_matches) else len(full_text)
        block = full_text[start:end]

        code = hm.group('code').strip()
        name = re.sub(r'\s+', ' ', hm.group('name')).strip()
        semester = semester_map.get(code)

        subject = Subject(subject_code=code, subject_name=name, semester=semester)

        # --- split off TEXT BOOKS / REFERENCE BOOKS / COURSE OUTCOMES tail ---
        tb_m = TEXTBOOKS_HDR_RE.search(block)
        rb_m = REFBOOKS_HDR_RE.search(block)
        co_m = COURSE_OUTCOMES_HDR_RE.search(block)
        map_m = MAPPING_HDR_RE.search(block)

        units_end = min([m.start() for m in [tb_m, rb_m, co_m, map_m] if m] or [len(block)])
        units_block = block[:units_end]

        if tb_m:
            tb_end = rb_m.start() if rb_m else (co_m.start() if co_m else (map_m.start() if map_m else len(block)))
            subject.text_books = split_citations(block[tb_m.end():tb_end], "textbook")
        if rb_m:
            rb_end = co_m.start() if co_m else (map_m.start() if map_m else len(block))
            subject.reference_books = split_citations(block[rb_m.end():rb_end], "reference")

        # --- parse units/modules ---
        unit_matches = list(UNIT_RE.finditer(units_block))
        for uidx, um in enumerate(unit_matches):
            roman, title, hours = um.group(1), um.group(2).strip(), um.group(3)
            body_start = um.end()
            body_end = unit_matches[uidx + 1].start() if uidx + 1 < len(unit_matches) else len(units_block)
            body = units_block[body_start:body_end]

            module = Module(
                module_no=ROMAN_MAP.get(roman, uidx + 1),
                module_title=re.sub(r'\s+', ' ', title),
                hours=hours,
                raw_text=re.sub(r'\s+', ' ', body).strip(),
            )
            module.topics = split_topics(body)
            subject.modules.append(module)

        if subject.modules:  # only keep subjects where we actually found units
            subjects.append(subject)

    return subjects


# --------------------------------------------------------------------------- #
# CLI / quick test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "AI_DS.pdf"
    filter_code = sys.argv[2] if len(sys.argv) > 2 else None

    subs = parse_syllabus(path)
    print(f"Parsed {len(subs)} subjects")

    to_show = [s for s in subs if s.subject_code == filter_code] if filter_code else subs[:2]
    for s in to_show:
        print(f"\n{s.subject_code} - {s.subject_name} (Sem {s.semester})")
        for m in s.modules:
            print(f"  Module {m.module_no}: {m.module_title} ({m.hours} hrs) "
                  f"-- {len(m.topics)} topics")
            for t in m.topics:               # full list -- no slicing
                print(f"    Topic: {t.topic}")
                for sub in t.subtopics:       # full list -- no slicing
                    print(f"       - {sub}")
        for b in s.text_books:
            print(f"  [Textbook] {b.author} | {b.title} | {b.edition} | {b.publisher} | {b.year}")
        for b in s.reference_books:
            print(f"  [Reference] {b.author} | {b.title} | {b.edition} | {b.publisher} | {b.year}")

    if not filter_code:
        print(f"\n(Showing first 2 of {len(subs)} subjects. Pass a subject code as "
              f"the 2nd argument to inspect one fully, e.g.:\n"
              f"  python syllabus_parser.py {path} HS4101)")