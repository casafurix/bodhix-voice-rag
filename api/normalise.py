"""Normalise stage — NFC unicode normalisation + language id.
See docs/06-harness.md DAG, stage `normalise`.
"""

from __future__ import annotations

import unicodedata

import regex
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # deterministic langdetect output

_ASCII_ONLY_RE = regex.compile(r"^[\x00-\x7F]*$")


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def canonical_language_code(language: str) -> str:
    """Convert provider regional codes such as ``en-IN`` to ``en``."""
    return language.strip().lower().replace("_", "-").split("-", 1)[0]


def detect_language(text: str, lang_hint: str | None = None) -> tuple[str, float]:
    if lang_hint:
        return canonical_language_code(lang_hint), 1.0
    try:
        candidates = detect_langs(text)
        top = candidates[0]
        lang, prob = top.lang, top.prob
    except Exception:
        return "unknown", 0.0

    # langdetect is unreliable on short strings and, verified against real
    # MSMARCO-XI queries (bench/run_latency.py's over-refusal findings,
    # docs/13-build-status.md), regularly misclassifies plain English
    # queries as other Latin-script languages -- e.g. "defination
    # arbitrary" and "does delta fly to bangalore" both came back with a
    # code outside our supported set. None of our other four languages
    # (hi/bn/ta/mr) use Latin script at all, so a non-English guess on
    # ASCII-only text can only be either langdetect noise on real English,
    # or a genuinely different Latin-script language we don't support
    # anyway (out of this project's scope either way) -- biasing ASCII-only
    # text toward "en" trades a small, accepted risk on that second case
    # for fixing the much more common first one.
    if lang != "en" and _ASCII_ONLY_RE.match(text):
        return "en", prob
    return lang, prob
