"""Generate bench/queries.jsonl — the committed query set every benchmark
runs over. Deterministic (seeded), derived from the SAME raw rows that
built the index, so every query has real relevance labels:

- in-domain queries: the dataset's own `query` (per language) + `Eng_Query`,
  with relevant doc ids = docs of that query_id carrying is_selected=true.
  These power both run_latency.py and run_retrieval.py's ablation.
- out-of-domain queries: a fixed hand-written list of questions on topics
  absent from the corpus, for measuring over-refusal rate.

Usage:
    uv run --extra dev python -m bench.make_queries [rows_per_language]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ingest.stream_corpus import LANGUAGE_FILES, load_language_rows

OUT = Path("bench/queries.jsonl")
MIN_TEXT_LEN = 5

OUT_OF_DOMAIN = [
    {"lang": "en", "query": "what is the capital of India"},
    {"lang": "en", "query": "how do I bake a chocolate cake"},
    {"lang": "en", "query": "who won the football world cup final"},
    {"lang": "en", "query": "what is the boiling point of water"},
    {"lang": "en", "query": "best restaurants in Paris"},
    {"lang": "hi", "query": "भारत की राजधानी क्या है"},
    {"lang": "hi", "query": "चॉकलेट केक कैसे बनाते हैं"},
    {"lang": "hi", "query": "फुटबॉल विश्व कप किसने जीता"},
    {"lang": "bn", "query": "ভারতের রাজধানী কী"},
    {"lang": "bn", "query": "চকলেট কেক কীভাবে বানায়"},
    {"lang": "ta", "query": "இந்தியாவின் தலைநகர் எது"},
    {"lang": "ta", "query": "சாக்லேட் கேக் எப்படி செய்வது"},
    {"lang": "mr", "query": "भारताची राजधानी काय आहे"},
    {"lang": "mr", "query": "चॉकलेट केक कसा बनवायचा"},
    {"lang": "en", "query": "how to learn japanese quickly"},
    {"lang": "en", "query": "what causes earthquakes"},
    {"lang": "en", "query": "symptoms of vitamin d deficiency"},
    {"lang": "hi", "query": "स्टॉक मार्केट में निवेश कैसे करें"},
    {"lang": "bn", "query": "ভূমিকম্প কেন হয়"},
    {"lang": "ta", "query": "புகைப்படம் எடுப்பது எப்படி"},
]


def main() -> None:
    rows_per_language = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    records: list[dict] = []
    seen_en_qids: set[str] = set()  # Eng_Query is the same source text across every
    # language file (row order is aligned) - dedupe so English isn't 5x overrepresented.

    for lang in LANGUAGE_FILES:
        rows = load_language_rows(lang, limit_rows=rows_per_language)
        kept = 0
        for row in rows:
            qid = str(row["query_id"])
            passages = row["passages"]
            selected_eng = [
                f"en/{qid}/p{i}"
                for i, sel in enumerate(passages["is_selected"])
                if sel and passages["English_passages"][i].strip()
            ]
            selected_indic = [
                f"{lang}/{qid}/p{i}"
                for i, sel in enumerate(passages["is_selected"])
                if sel and passages["Translated_passages"][i].strip()
            ]
            if not selected_eng and not selected_indic:
                continue  # no positive label -> useless for the ablation

            # This language's own translated query - always distinct per (lang, qid).
            if row["query"] and len(row["query"].strip()) >= MIN_TEXT_LEN:
                records.append(
                    {
                        "qid": qid,
                        "lang": lang,
                        "query": row["query"],
                        "relevant_prefixes": [f"en/{qid}/", f"{lang}/{qid}/"],
                        "n_selected": len(selected_eng) + len(selected_indic),
                        "domain": "in",
                    }
                )
                kept += 1

            # The shared English query - emitted once per qid, first file wins.
            eng_query = row.get("Eng_Query")
            if qid not in seen_en_qids and eng_query and len(eng_query.strip()) >= MIN_TEXT_LEN:
                records.append(
                    {
                        "qid": qid,
                        "lang": "en",
                        "query": eng_query,
                        "relevant_prefixes": [f"en/{qid}/"],
                        "n_selected": len(selected_eng),
                        "domain": "in",
                    }
                )
                seen_en_qids.add(qid)
                kept += 1
        print(f"[queries] {lang}: {kept} usable queries from {rows_per_language} rows")

    for i, ood in enumerate(OUT_OF_DOMAIN):
        records.append(
            {
                "qid": f"ood-{i}",
                "lang": ood["lang"],
                "query": ood["query"],
                "relevant_prefixes": [],
                "n_selected": 0,
                "domain": "out",
            }
        )

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    n_in = sum(1 for r in records if r["domain"] == "in")
    print(f"[queries] wrote {len(records)} ({n_in} in-domain, "
          f"{len(records) - n_in} out-of-domain) to {OUT}")


if __name__ == "__main__":
    main()
