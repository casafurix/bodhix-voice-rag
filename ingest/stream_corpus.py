"""Pull the MSMARCO-XI validation files for our language set. See
docs/02-dataset.md, "Corpus construction plan".

Scope note vs. the doc: "stream via row-groups over HTTP range requests"
is the doc's plan for avoiding a 55GB download. In practice we only ever
need validation/hinval.parquet + validation/benval.parquet (~460MB each,
per the dataset's own file listing) — English text rides along for free
inside every row (Eng_Query/Eng_Answer/English_passages), so we don't
need a separate English file at all. Downloading two ~460MB files (not
55GB) and reading them locally with polars is simpler than building a
row-group HTTP-range streamer for a corpus this small, and is well within
"never hold more than a few GB on disk."
"""

from __future__ import annotations

import polars as pl
from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/MSMARCO-XI"

# Indic languages we ingest natively. English is not listed here — it is
# derived from every row's Eng_Query/Eng_Answer/English_passages columns,
# not from its own file. See docs/02-dataset.md on why Telugu is
# validation-only and excluded from our language set entirely.
LANGUAGE_FILES: dict[str, str] = {
    "hi": "validation/hinval.parquet",
    "bn": "validation/benval.parquet",
}

COLUMNS = [
    "query",
    "Answer",
    "query_id",
    "query_type",
    "passages",
    "Eng_Query",
    "Eng_Answer",
    "source_lang",
    "target_lang",
]


def download_language_file(lang: str) -> str:
    if lang not in LANGUAGE_FILES:
        raise ValueError(f"no validation file configured for language '{lang}'")
    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=LANGUAGE_FILES[lang])


def load_language_rows(lang: str, limit_rows: int | None = None) -> list[dict]:
    """Returns raw row dicts (still one row = one query + up to 10 passages,
    not yet exploded). Column-projected before collect() so we never
    materialise the `meta` struct or anything else we don't use.
    """
    path = download_language_file(lang)
    lf = pl.scan_parquet(path).select(COLUMNS)
    if limit_rows is not None:
        lf = lf.head(limit_rows)
    return lf.collect().to_dicts()
