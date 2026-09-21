"""
topic_mapper.py
----------------
Given:
  - a syllabus topic/sub-topic (short phrase, e.g. "Torsion pendulum: theory
    and experiment")
  - a book's per-page text index (from book_indexer.py)

...finds the most likely page_start / page_end range in the book that
covers that topic, and a short coverage_notes string ("primary",
"supplementary", "brief mention", etc).

Two-stage approach (cheap-first, LLM-second):
  1. CANDIDATE RETRIEVAL (no LLM, fast): score every page by fuzzy text
     similarity + keyword overlap against the topic phrase. Take the
     top-N candidate pages.
  2. LLM ADJUDICATION: send the topic + the candidate pages' text excerpts
     to a local Ollama model (or any OpenAI-compatible chat endpoint) and
     ask it to pick the actual contiguous page range, chapter number (if
     inferable), and a coverage note. The LLM only ever sees a handful of
     short excerpts, not the whole book, so this stays fast and cheap even
     for 500+ page scanned textbooks.

Swap LLM backend by editing `call_llm()` below -- Ollama is the default
because it's free/local and needs no API key, but the OpenAI-compatible
branch is included for e.g. Anthropic/OpenAI API users.
"""

import json
import re
from typing import List, Dict, Optional

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False


# --------------------------------------------------------------------------- #
# Stage 1: candidate retrieval (no LLM)
# --------------------------------------------------------------------------- #

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "using", "use", "introduction", "concepts", "concept", "basic", "basics",
}


def _keywords(phrase: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", phrase.lower())
    return [w for w in words if w not in STOPWORDS]


def score_page(topic: str, page_text: str) -> float:
    if not page_text.strip():
        return 0.0
    kw = _keywords(topic)
    text_lower = page_text.lower()
    kw_hits = sum(1 for k in kw if k in text_lower)
    kw_score = kw_hits / max(len(kw), 1)

    fuzzy_score = 0.0
    if HAVE_RAPIDFUZZ:
        # partial_ratio handles the fact that a page is much longer than the topic phrase
        fuzzy_score = fuzz.partial_ratio(topic.lower(), text_lower[:4000]) / 100.0

    return 0.6 * kw_hits + 0.4 * fuzzy_score + 0.05 * kw_score


def rank_candidate_pages(topic: str, index: Dict, top_k: int = 8) -> List[Dict]:
    scored = []
    for page in index["pages"]:
        s = score_page(topic, page["text"])
        if s > 0:
            scored.append((s, page))
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]
    return [
        {"page_num": p["page_num"], "score": round(s, 2), "excerpt": p["text"][:500]}
        for s, p in top
    ]


# --------------------------------------------------------------------------- #
# Stage 2: LLM adjudication
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an assistant that maps a college syllabus topic to \
the exact page range in a specific textbook where that topic is covered, \
based ONLY on the candidate page excerpts provided (you cannot see the rest \
of the book). Respond with STRICT JSON only, no markdown, no commentary, \
matching this schema:

{
  "found": true | false,
  "page_start": <int or null>,
  "page_end": <int or null>,
  "chapter": "<string/int or empty>",
  "coverage": "primary" | "supplementary" | "brief mention" | "not found",
  "notes": "<concise note or observation, e.g. 'Simpler explanation of topic', 'In-depth derivation and proofs', etc., or empty string>"
}

Rules:
- Only use page numbers that appear in the candidate list given to you.
- If multiple candidate pages are contiguous and relevant, span page_start..page_end across them.
- If nothing in the candidates actually covers the topic, set "found": false, "coverage": "not found", "notes": "".
- "coverage": "primary" = the topic is explained in depth here; "supplementary" = touched on briefly / side reference / simpler introductory treatment.
- "chapter": provide the chapter number (e.g. 1, 2, 3) or short chapter title from excerpts.
- "notes": short comment describing emphasis or context (or empty string for standard primary coverage).
"""


def _build_user_prompt(topic: str, subject_name: str, module_title: str,
                        book_title: str, candidates: List[Dict]) -> str:
    cand_text = "\n\n".join(
        f"[Page {c['page_num']}] (match score {c['score']})\n{c['excerpt']}"
        for c in candidates
    )
    return f"""Subject: {subject_name}
Module: {module_title}
Book: {book_title}
Topic to locate: "{topic}"

Candidate pages (only these page numbers are valid to use):
{cand_text}

Return the JSON now."""


def call_llm_openai_compatible(system_prompt: str, user_prompt: str,
                                model: str = "gpt-4o-mini",
                                base_url: Optional[str] = None,
                                api_key_env: str = "OPENAI_API_KEY") -> str:
    """Generic backend for anyone who'd rather use an OpenAI-compatible API
    (OpenAI, Azure OpenAI, or any local server exposing the same schema,
    e.g. vLLM / LM Studio) instead of Ollama/Gemini directly."""
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get(api_key_env, "not-needed"), base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def call_llm_omniroute(system_prompt: str, user_prompt: str,
                        model: str = "gemini-2.5-flash",
                        base_url_env: str = "OMNIROUTE_BASE_URL",
                        api_key_env: str = "OMNIROUTE_API_KEY") -> str:
    """OmniRoute backend -- an open-source AI gateway that fronts 290+
    providers (Gemini, free tiers, local Ollama, etc.) behind a single
    OpenAI-compatible endpoint, with automatic fallback if a provider hits
    a rate limit or goes down. Since it's OpenAI-compatible, this just
    reuses call_llm_openai_compatible() pointed at your OmniRoute instance.

    Setup:
      1. Run OmniRoute (e.g. `npx omniroute` or the desktop app) -- it comes
         up on http://localhost:20128/v1 by default.
      2. Open its dashboard, connect at least one provider (Gemini's free
         tier works well for this task), and copy the dashboard API key.
      3. export OMNIROUTE_BASE_URL="http://localhost:20128/v1"
         export OMNIROUTE_API_KEY="<dashboard key>"
      4. Run this pipeline with --backend omniroute --model <model-alias>
         (whatever model string your OmniRoute routing config expects --
         check its dashboard/docs; "auto" lets OmniRoute pick for you on
         some setups).

    The benefit over calling Gemini directly: if you burn through Gemini's
    free quota mid-run on a big book, OmniRoute automatically fails over to
    another connected provider instead of the pipeline erroring out.
    """
    import os
    base_url = os.environ.get(base_url_env, "http://localhost:20128/v1")
    return call_llm_openai_compatible(
        system_prompt, user_prompt, model=model,
        base_url=base_url, api_key_env=api_key_env,
    )


def call_llm_gemini(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash",
                     api_key_env: str = "GEMINI_API_KEY") -> str:
    """Primary/recommended backend. Needs: pip install google-genai
    and an API key from https://aistudio.google.com/apikey exported as
    GEMINI_API_KEY (or pass a different env var name)."""
    import os
    from google import genai
    from google.genai import types

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set the {api_key_env} environment variable with your Gemini API key.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return resp.text


def call_llm_ollama(system_prompt: str, user_prompt: str, model: str = "llama3.1") -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",  # ask ollama to constrain output to valid JSON
        options={"temperature": 0.0},
    )
    return resp["message"]["content"]


def map_topic_to_pages(topic: str, subject_name: str, module_title: str,
                        book_title: str, index: Dict,
                        backend: str = "gemini", model: str = "gemini-2.5-flash",
                        top_k: int = 8) -> Dict:
    """Full pipeline for one topic against one book. Returns a dict with
    page_start, page_end, chapter, coverage_notes (see SYSTEM_PROMPT schema)."""

    candidates = rank_candidate_pages(topic, index, top_k=top_k)
    if not candidates:
        return {"found": False, "page_start": None, "page_end": None,
                "chapter": "", "coverage_notes": "not found"}

    user_prompt = _build_user_prompt(topic, subject_name, module_title, book_title, candidates)

    try:
        if backend == "gemini":
            raw = call_llm_gemini(SYSTEM_PROMPT, user_prompt, model=model)
        elif backend == "ollama":
            raw = call_llm_ollama(SYSTEM_PROMPT, user_prompt, model=model)
        elif backend == "openai":
            raw = call_llm_openai_compatible(SYSTEM_PROMPT, user_prompt, model=model)
        elif backend == "omniroute":
            raw = call_llm_omniroute(SYSTEM_PROMPT, user_prompt, model=model)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        result = json.loads(raw)
    except Exception as e:
        # LLM unavailable / failed -> fall back to a pure heuristic answer
        # so the pipeline still produces usable (if less precise) output.
        best = candidates[0]
        result = {
            "found": True,
            "page_start": best["page_num"],
            "page_end": candidates[min(2, len(candidates) - 1)]["page_num"],
            "chapter": "",
            "coverage": "primary",
            "notes": f"heuristic match (LLM unavailable: {e})",
        }

    # Clean up chapter format (e.g. extract chapter number "2" from "Chapter 2: ...")
    raw_chap = str(result.get("chapter", "") or "").strip()
    m = re.match(r"^Chapter\s+(\d+|[IVXLCDM]+)\b(?:\s*[:.\-]\s*.*)?$", raw_chap, re.IGNORECASE)
    if m:
        result["chapter"] = m.group(1)
    else:
        result["chapter"] = raw_chap

    # Normalize coverage field
    cov = result.get("coverage") or result.get("coverage_type") or result.get("coverage_notes") or "primary"
    result["coverage"] = cov
    result["notes"] = result.get("notes") or ""

    # normalize page_start <= page_end
    if result.get("page_start") and result.get("page_end"):
        if result["page_end"] < result["page_start"]:
            result["page_start"], result["page_end"] = result["page_end"], result["page_start"]

    return result
