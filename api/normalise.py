"""Normalise stage — NFC unicode normalisation + language id.
See docs/06-harness.md DAG, stage `normalise`.
"""

from __future__ import annotations

import unicodedata

from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # deterministic langdetect output


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
        return top.lang, top.prob
    except Exception:
        return "unknown", 0.0
