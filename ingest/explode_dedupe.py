"""Explode raw rows into deduped PassageDoc instances across en/hi/bn.
See docs/02-dataset.md, "Step 2 — Explode and deduplicate".

Each row's `passages` struct has up to 10 (Translated_passages,
English_passages, is_selected) triples. For a Hindi row we emit:
- one Hindi PassageDoc per passage, twinned to the English passage at the
  same index (twin_doc_id/twin_text), free_question = the row's own
  (Hindi) `query`
- one English PassageDoc per passage, twinned back to the Hindi one,
  free_question = `Eng_Query`

Row order is identical across every per-language file in this dataset
(verified by hand: hinval.parquet and benval.parquet's first N rows carry
the same query_ids in the same order) — but we don't rely on that beyond
it being a nice performance property. Processing hi and bn files
independently and re-emitting English docs from both is deliberately
simple and safe: exact-duplicate English passages collapse via the
blake3 dedup below regardless of whether the alignment assumption holds.

Dedup key: blake3(NFC-normalised, whitespace-collapsed text), lowercased
for Latin script only — case is meaningless in Devanagari/Bengali, per
the doc. First occurrence wins; we track the set of query_ids that
reference a passage as `ref_count`, a popularity signal the doc flags as
a possible (unvalidated) retrieval prior — recorded but not used to rank
anything yet.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import blake3

from api.retrieval.chunkers.base import PassageDoc

_SCRIPT_BY_LANG = {"hi": "Deva", "bn": "Beng", "en": "Latn"}


def _normalise_for_dedup(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    if lang == "en":
        text = text.lower()
    return text


def _dedup_key(text: str, lang: str) -> str:
    return blake3.blake3(_normalise_for_dedup(text, lang).encode("utf-8")).hexdigest()


@dataclass
class DedupIndex:
    """Tracks which normalised texts we've already emitted a doc for, and
    how many distinct queries reference each one.
    """

    seen: dict[str, str] = field(default_factory=dict)  # dedup_key -> doc_id kept
    ref_counts: dict[str, int] = field(default_factory=dict)  # doc_id -> count

    def register(self, text: str, lang: str, candidate_doc_id: str) -> tuple[str, bool]:
        """Returns (doc_id_to_use, is_new). If this text was already seen,
        returns the doc_id of the first occurrence and increments its
        ref_count instead of creating a duplicate.
        """
        key = _dedup_key(text, lang)
        if key in self.seen:
            existing = self.seen[key]
            self.ref_counts[existing] = self.ref_counts.get(existing, 1) + 1
            return existing, False
        self.seen[key] = candidate_doc_id
        self.ref_counts[candidate_doc_id] = 1
        return candidate_doc_id, True


def explode_rows(rows: list[dict], indic_lang: str, dedup: DedupIndex) -> list[PassageDoc]:
    """`rows` are raw dicts from stream_corpus.load_language_rows() for one
    Indic language file. Returns deduped PassageDoc instances for both
    that Indic language and English (derived from the same rows).
    """
    docs: list[PassageDoc] = []

    for row in rows:
        query_id = row["query_id"]
        query_type = row["query_type"]
        indic_query = row["query"]
        eng_query = row["Eng_Query"]
        passages = row["passages"]
        translated = passages["Translated_passages"]
        english = passages["English_passages"]
        is_selected = passages["is_selected"]

        for i in range(len(translated)):
            indic_text = translated[i]
            eng_text = english[i]
            selected = bool(is_selected[i])

            indic_candidate_id = f"{indic_lang}/{query_id}/p{i}"
            eng_candidate_id = f"en/{query_id}/p{i}"

            indic_doc_id, indic_is_new = dedup.register(indic_text, indic_lang, indic_candidate_id)
            eng_doc_id, eng_is_new = dedup.register(eng_text, "en", eng_candidate_id)

            if indic_is_new:
                docs.append(
                    PassageDoc(
                        doc_id=indic_doc_id,
                        text=indic_text,
                        language=indic_lang,
                        script=_SCRIPT_BY_LANG[indic_lang],
                        query_id=query_id,
                        query_type=query_type,
                        is_selected=selected,
                        twin_doc_id=eng_doc_id,
                        twin_text=eng_text,
                        free_question=indic_query,
                    )
                )
            if eng_is_new:
                docs.append(
                    PassageDoc(
                        doc_id=eng_doc_id,
                        text=eng_text,
                        language="en",
                        script=_SCRIPT_BY_LANG["en"],
                        query_id=query_id,
                        query_type=query_type,
                        is_selected=selected,
                        twin_doc_id=indic_doc_id,
                        twin_text=indic_text,
                        free_question=eng_query,
                    )
                )

    return docs
