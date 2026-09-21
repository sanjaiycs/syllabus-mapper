"""
main.py
-------
Orchestrates the full pipeline:

  1. Parse the syllabus PDF -> subjects -> modules -> topics + book citations
     (syllabus_parser.py)
  2. For each subject, load the ACTUAL local book file(s) you own, as
     declared in book_files.json (see below), and build a per-page text
     index -- OCR automatically kicks in for scanned pages
     (book_indexer.py)
  3. For each topic, find the page range in each book that covers it, using
     fuzzy-match candidate retrieval + LLM adjudication
     (topic_mapper.py)
  4. Write everything out as a CSV with exactly the columns from your
     reference spreadsheet:

     id, subject_code, subject_name, module_no, module_title, topic,
     sub_topic, book_title, author, edition, publisher, year, book_type,
     chapter, page_start, page_end, coverage_notes


WHY book_files.json?
---------------------
The syllabus PDF only gives you *citations* ("Reema Thareja, Data
Structures, Oxford, 2nd ed, 2014") -- it doesn't know where the physical
book file lives on your machine, and citation text is too unreliable to
auto-match to a filename. So you tell the pipeline once, per subject code,
which local file corresponds to which citation. Example:

{
  "DS": [
    {
      "file": "books/data_structures_reema_thareja.pdf",
      "title": "Data Structures",
      "author": "Reema Thareja",
      "edition": "2nd",
      "publisher": "Oxford",
      "year": "2014",
      "book_type": "reference"
    },
    {
      "file": "books/clrs_3rd_edition.pdf",
      "title": "Introduction to Algorithms",
      "author": "Cormen, L",
      "edition": "3rd",
      "publisher": "MIT Press",
      "year": "2009",
      "book_type": "textbook"
    }
  ]
}

("DS" here is whatever short code you want to key subjects by -- easiest is
to just use the subject_code from the syllabus, e.g. "DS4301".)


USAGE
-----
    python main.py \\
        --syllabus AI_DS.pdf \\
        --book-files book_files.json \\
        --subjects DS4301 \\
        --out mapped_topics.csv \\
        --backend ollama --model llama3.1

Omit --subjects to process every subject found in the syllabus (slow --
each topic x book pair is one LLM call).
"""

import argparse
import csv
import json
import os
import re
import sys

from syllabus_parser import parse_syllabus
from book_indexer import build_book_index
from topic_mapper import map_topic_to_pages

CSV_COLUMNS = [
    "id", "subject_code", "subject_name", "module_no", "module_title",
    "topic", "sub_topic", "book_title", "author", "edition", "publisher",
    "year", "book_type", "chapter", "page_start", "page_end", "coverage", "notes",
]


def load_env(path=".env"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


def load_book_files(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_id(subject_code, module_no, topic_idx, sub_idx=None):
    # e.g. "DS_M1_T1" for a topic-level row, "DS_M1_T1_S2" for a sub-topic row
    prefix = re.sub(r"[^A-Za-z]", "", subject_code)[:2].upper() or "SJ"
    base = f"{prefix}_M{module_no}_T{topic_idx}"
    return f"{base}_S{sub_idx}" if sub_idx else base


def run(syllabus_path, book_files_path, out_path, subject_filter=None,
        backend="gemini", model="gemini-2.5-flash", top_k=8, printed_page_offset=0,
        max_topics_per_module=None, max_books_per_subject=None):

    load_env()
    subjects = parse_syllabus(syllabus_path)
    book_files = load_book_files(book_files_path)

    if subject_filter:
        subjects = [s for s in subjects if s.subject_code in subject_filter]

    if not subjects:
        print("No matching subjects found. Check --subjects / syllabus parsing.")
        return

    row_count = 0
    index_cache = {}  # file path -> built index (avoid re-indexing same book)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()

        for subject in subjects:
            books = book_files.get(subject.subject_code)
            if not books:
                print(f"[skip] No book_files.json entry for {subject.subject_code} "
                      f"({subject.subject_name}) -- add one to map topics to pages.")
                continue

            if max_books_per_subject:
                books = books[:max_books_per_subject]

            print(f"\n=== {subject.subject_code} - {subject.subject_name} ===")

            for book in books:
                book_path = book["file"]
                if not os.path.exists(book_path):
                    print(f"  [!] Book file not found: {book_path} -- skipping.")
                    continue

                if book_path not in index_cache:
                    print(f"  Indexing {book_path} ...")
                    index_cache[book_path] = build_book_index(book_path)
                index = index_cache[book_path]

                for module in subject.modules:
                    topic_items = module.topics
                    if max_topics_per_module:
                        topic_items = topic_items[:max_topics_per_module]

                    for t_idx, topic_item in enumerate(topic_items, start=1):
                        # Build the list of (sub_idx, query_text, topic_label, subtopic_label)
                        # to map -- one entry per sub-topic if any exist, else one
                        # entry for the topic itself.
                        if topic_item.subtopics:
                            query_units = [
                                (s_idx, subtopic, topic_item.topic, subtopic)
                                for s_idx, subtopic in enumerate(topic_item.subtopics, start=1)
                            ]
                        else:
                            query_units = [(None, topic_item.topic, topic_item.topic, "")]

                        for sub_idx, query_text, topic_label, subtopic_label in query_units:
                            tag = f"T{t_idx}" + (f"_S{sub_idx}" if sub_idx else "")
                            print(f"  [{book['title']}] M{module.module_no} "
                                  f"{tag}: {query_text[:60]}")

                            result = map_topic_to_pages(
                                topic=query_text,
                                subject_name=subject.subject_name,
                                module_title=module.module_title,
                                book_title=book["title"],
                                index=index,
                                backend=backend,
                                model=model,
                                top_k=top_k,
                            )

                            if not result.get("found", True) and result.get("page_start") is None:
                                continue  # this book doesn't cover this topic/subtopic -- skip row

                            page_start = result.get("page_start")
                            page_end = result.get("page_end")
                            if page_start is not None and printed_page_offset:
                                page_start += printed_page_offset
                            if page_end is not None and printed_page_offset:
                                page_end += printed_page_offset

                            row = {
                                "id": make_id(subject.subject_code, module.module_no, t_idx, sub_idx),
                                "subject_code": subject.subject_code,
                                "subject_name": subject.subject_name,
                                "module_no": module.module_no,
                                "module_title": module.module_title,
                                "topic": topic_label,
                                "sub_topic": subtopic_label,
                                "book_title": book["title"],
                                "author": book.get("author", ""),
                                "edition": book.get("edition", ""),
                                "publisher": book.get("publisher", ""),
                                "year": book.get("year", ""),
                                "book_type": book.get("book_type", ""),
                                "chapter": result.get("chapter", ""),
                                "page_start": page_start,
                                "page_end": page_end,
                                "coverage": result.get("coverage", "primary"),
                                "notes": result.get("notes", ""),
                            }
                            writer.writerow(row)
                            f.flush()
                            row_count += 1

    print(f"\nDone. {row_count} rows written -> {out_path}")


if __name__ == "__main__":
    default_syllabus = "AI_DS.pdf" if os.path.exists("AI_DS.pdf") else None
    default_book_files = "book_files.json" if os.path.exists("book_files.json") else None

    ap = argparse.ArgumentParser(description="Map syllabus topics to textbook page ranges.")
    ap.add_argument("--syllabus", default=default_syllabus, required=(default_syllabus is None),
                    help="Path to syllabus PDF (default: AI_DS.pdf)")
    ap.add_argument("--book-files", default=default_book_files, required=(default_book_files is None),
                    help="Path to book_files.json (default: book_files.json)")
    ap.add_argument("--out", default="mapped_topics.csv", help="Output CSV path (default: mapped_topics.csv)")
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="Subject codes to process (default: all subjects in book_files.json)")
    default_backend = "gemini"
    default_model = "gemini-2.5-flash"
    load_env()
    if os.environ.get("OMNIROUTE_API_KEY"):
        default_backend = "omniroute"
        default_model = "auto/best-coding"
    elif os.environ.get("OPENAI_API_KEY"):
        default_backend = "openai"
        default_model = "gpt-4o-mini"

    ap.add_argument("--backend", choices=["gemini", "ollama", "openai", "omniroute"], default=default_backend)
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--top-k", type=int, default=8, help="Candidate pages sent to the LLM per topic")
    ap.add_argument("--printed-page-offset", type=int, default=0,
                     help="Add this to physical PDF page numbers to match the book's printed page numbers")
    ap.add_argument("--max-topics-per-module", type=int, default=None,
                     help="Cap topics per module (useful for a quick test run)")
    ap.add_argument("--max-books-per-subject", type=int, default=None,
                     help="Cap books processed per subject (useful for testing on just 1 book)")
    args = ap.parse_args()

    run(
        syllabus_path=args.syllabus,
        book_files_path=args.book_files,
        out_path=args.out,
        subject_filter=args.subjects,
        backend=args.backend,
        model=args.model,
        top_k=args.top_k,
        printed_page_offset=args.printed_page_offset,
        max_topics_per_module=args.max_topics_per_module,
        max_books_per_subject=args.max_books_per_subject,
    )
