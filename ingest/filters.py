"""Quality filters — see docs/02-dataset.md, "Step 3 — Quality filtering".

MVP scope: length and script-purity only. The doc's table also lists
twin-length-ratio, twin-embedding-cosine, near-duplicate (MinHash) and
boilerplate-pattern filters — cut for the MVP (they need an embedding pass
or a similarity index at ingest time, which is real extra engineering for
a filter whose benefit the doc itself says must be measured before it
ships: "if it does not improve nDCG, it comes out"). Rejection counts are
returned so the gap is visible, not silent.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

MIN_CHARS = 20
MAX_CHARS = 4000
MIN_SCRIPT_PURITY = 0.6  # fraction of alphabetic chars that must match the expected script

# Unicode block ranges used for the script-purity check, per language.
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "hi": [(0x0900, 0x097F)],  # Devanagari
    "bn": [(0x0980, 0x09FF)],  # Bengali
    "ta": [(0x0B80, 0x0BFF)],  # Tamil
    "mr": [(0x0A80, 0x0AFF)],  # Devanagari (Marathi)
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],  # Latin A-Z / a-z
}


@dataclass
class FilterCounts:
    total: int = 0
    kept: int = 0
    rejected_length: int = 0
    rejected_script: int = 0
    detail: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "rejected_length": self.rejected_length,
            "rejected_script": self.rejected_script,
        }


def _script_purity(text: str, lang: str) -> float:
    ranges = _SCRIPT_RANGES.get(lang)
    if not ranges:
        return 1.0  # unknown language: don't reject on a check we can't perform
    alphabetic = [c for c in text if c.isalpha()]
    if not alphabetic:
        return 1.0  # no alphabetic chars (e.g. pure numeric) — not a script failure
    matches = sum(1 for c in alphabetic if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return matches / len(alphabetic)


def passes_filters(text: str, lang: str, counts: FilterCounts) -> bool:
    counts.total += 1
    text = unicodedata.normalize("NFC", text)

    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        counts.rejected_length += 1
        return False

    if _script_purity(text, lang) < MIN_SCRIPT_PURITY:
        counts.rejected_script += 1
        return False

    counts.kept += 1
    return True
