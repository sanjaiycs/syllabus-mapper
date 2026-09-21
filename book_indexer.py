"""
book_indexer.py
----------------
Builds a per-page text index for a textbook, whether it's a normal
(digitally-typeset) PDF or a scanned/image-only PDF.

Strategy per page:
  1. Try direct text extraction (PyMuPDF if installed, else pdfplumber).
  2. If the extracted text is suspiciously short (< MIN_CHARS_FOR_TEXT_PAGE),
     assume it's a scanned/image page and OCR it instead
     (pdf2image -> PIL image -> pytesseract).

The result is cached to a JSON file next to the PDF (<name>.index.json) so
you don't have to re-OCR a 500-page scanned book every run.

Output structure:
    {
      "pdf_path": "...",
      "book_title": "...",          # filled in by you / metadata.json
      "num_pages": 350,
      "pages": [
         {"page_num": 1, "text": "...", "source": "text" | "ocr"},
         ...
      ]
    }

Page numbering here is the *physical* PDF page index (1-based), which is
usually offset from the *printed* page number in the book (front matter,
prefaces etc). See `printed_page_offset` in main.py / metadata.json to
correct for that when producing the final report.
"""

import os
import json
import re

MIN_CHARS_FOR_TEXT_PAGE = 30  # below this, treat page as scanned/image


def _extract_text_pages_pymupdf(pdf_path):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc], len(doc)


def _extract_text_pages_pdfplumber(pdf_path):
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]
        return texts, len(pdf.pages)


def _ocr_page(pdf_path, page_index, dpi=200):
    """OCR a single page (0-indexed) of a PDF via pdf2image + pytesseract."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=page_index + 1, last_page=page_index + 1
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def build_book_index(pdf_path, cache=True, ocr_dpi=200, verbose=True):
    cache_path = os.path.splitext(pdf_path)[0] + ".index.json"
    if cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        texts, num_pages = _extract_text_pages_pymupdf(pdf_path)
    except ImportError:
        texts, num_pages = _extract_text_pages_pdfplumber(pdf_path)

    pages = []
    for i in range(num_pages):
        text = texts[i] or ""
        source = "text"
        if len(text.strip()) < MIN_CHARS_FOR_TEXT_PAGE:
            if verbose:
                print(f"  [OCR] page {i+1}/{num_pages} looks scanned, running OCR...")
            try:
                text = _ocr_page(pdf_path, i, dpi=ocr_dpi)
                source = "ocr"
            except Exception as e:
                if verbose:
                    print(f"    OCR failed on page {i+1}: {e}")
                source = "empty"
        pages.append({"page_num": i + 1, "text": text, "source": source})

    index = {"pdf_path": pdf_path, "num_pages": num_pages, "pages": pages}

    if cache:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(index, f)
        if verbose:
            print(f"  Cached index -> {cache_path}")

    return index


def find_keyword_hits(index, keyword, min_len=4):
    """Fast, non-LLM baseline: which pages literally mention this keyword/phrase
    (case-insensitive, whole-word-ish)."""
    if len(keyword.strip()) < min_len:
        return []
    pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
    hits = []
    for page in index["pages"]:
        if pattern.search(page["text"]):
            hits.append(page["page_num"])
    return hits


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    idx = build_book_index(path)
    print(f"Indexed {idx['num_pages']} pages from {path}")
    text_pages = sum(1 for p in idx["pages"] if p["source"] == "text")
    ocr_pages = sum(1 for p in idx["pages"] if p["source"] == "ocr")
    print(f"  {text_pages} text pages, {ocr_pages} OCR pages")
