"""The six chunking strategies shipped in the MVP — see docs/03-chunking.md,
"Minimum viable scope" (tiers 1 + 2 + the S1 control = exactly six).

S4, S6, S7, S8, S12 are cut per docs/11-roadmap.md descoping order.
"""

STRATEGY_IDS = [
    "s1_fixed",
    "s2_passage_native",
    "s3_sentence_window",
    "s5_parent_child",
    "s9_doc2query",
    "s10_crosslingual_twin",
]
