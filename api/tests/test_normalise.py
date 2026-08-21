from api.normalise import detect_language, normalise_text


def test_normalise_collapses_whitespace():
    assert normalise_text("hello   world\n\tfoo") == "hello world foo"


def test_normalise_strips_leading_trailing_whitespace():
    assert normalise_text("  hello  ") == "hello"


def test_detect_language_uses_hint_when_given():
    lang, conf = detect_language("some text", lang_hint="hi")
    assert lang == "hi"
    assert conf == 1.0


def test_detect_language_canonicalises_regional_hint():
    lang, conf = detect_language("some text", lang_hint="en-IN")
    assert lang == "en"
    assert conf == 1.0


def test_detect_language_detects_english():
    lang, conf = detect_language("This is a clearly written English sentence about history.")
    assert lang == "en"
    assert conf > 0.5


def test_detect_language_handles_empty_text_gracefully():
    lang, conf = detect_language("")
    assert lang == "unknown"
    assert conf == 0.0


def test_detect_language_biases_short_ascii_queries_to_english():
    # Real MSMARCO-XI queries langdetect misclassified in bench/run_latency.py
    # -- both are unambiguously English, short-text ASCII noise otherwise
    # would have refused them with UNSUPPORTED_LANGUAGE.
    for query in ["defination arbitrary", "does delta fly to bangalore"]:
        lang, _ = detect_language(query)
        assert lang == "en", f"{query!r} should resolve to en, got {lang!r}"


def test_detect_language_does_not_override_indic_script(monkeypatch):
    # The ASCII-only fallback must never fire on non-Latin text -- Devanagari
    # is nowhere near the ASCII range, so this should be untouched regardless
    # of what langdetect returns.
    lang, _ = detect_language("यह एक हिंदी वाक्य है")
    assert lang == "hi"
