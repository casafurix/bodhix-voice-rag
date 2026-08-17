"""Input guardrails — see docs/07-guardrails.md.

Ordered cheapest-first. Runs on the normalised query/transcript. MVP scope:
checks 1, 2, 5, 6, 7 (empty/too-short, too-long, unsupported language, unsafe,
injection). Checks 3-4 (audio-duration sanity, ASR-confidence gibberish) are
voice-specific and apply once the Sarvam adapter is wired end-to-end; check 8
(PII redaction) is deferred — see the MVP scope cuts.
"""

from __future__ import annotations

import regex

from api.config import settings
from api.harness.stage import StageShortCircuit

MIN_CHARS = 3
MAX_CHARS = 512

# Deliberately small, curated, multilingual (en/hi/ta) — see docs/07-guardrails.md
# on why an English-only list is security theatre on an Indic pipeline.
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"you are now",
    r"pretend (that )?you are",
    r"पिछले निर्देश (भूल|छोड़)",
    r"इस निर्देश को अनदेखा",
    r"முந்தைய அறிவுரைகளை புறக்கணி",
]
_INJECTION_RE = regex.compile("|".join(_INJECTION_PATTERNS), regex.IGNORECASE)

# Deliberately narrow, curated — see docs/07-guardrails.md on coverage honesty.
_UNSAFE_PATTERNS = [
    r"\bhow to (make|build|synthesi[sz]e) (a )?(bomb|explosive|weapon)\b",
    r"\bself[- ]harm\b",
    r"\bsuicide (method|instructions)\b",
]
_UNSAFE_RE = regex.compile("|".join(_UNSAFE_PATTERNS), regex.IGNORECASE)


def check_length(text: str) -> None:
    if len(text.strip()) < MIN_CHARS:
        raise StageShortCircuit("NO_SPEECH", "query too short / empty")
    if len(text) > MAX_CHARS:
        raise StageShortCircuit("MALFORMED_QUERY", f"query exceeds {MAX_CHARS} chars")


def check_language(detected_lang: str) -> None:
    if detected_lang not in settings.language_list:
        raise StageShortCircuit(
            "UNSUPPORTED_LANGUAGE", f"'{detected_lang}' not in {settings.language_list}"
        )


def check_unsafe(text: str) -> None:
    if _UNSAFE_RE.search(text):
        raise StageShortCircuit("UNSAFE_INPUT", "matched unsafe-content pattern")


def check_injection(text: str) -> None:
    if _INJECTION_RE.search(text):
        raise StageShortCircuit("INJECTION_DETECTED", "matched prompt-injection pattern")


def run_guard_in(text: str, detected_lang: str) -> dict[str, bool]:
    """Runs all checks in cost order. Raises StageShortCircuit on the first
    hit; returns a pass/fail map for the trace if everything passes.
    """
    checks_passed: dict[str, bool] = {}

    check_length(text)
    checks_passed["length"] = True

    check_language(detected_lang)
    checks_passed["language"] = True

    check_unsafe(text)
    checks_passed["unsafe"] = True

    check_injection(text)
    checks_passed["injection"] = True

    return checks_passed
