from api.retrieval.text_utils import sentence_spans, split_sentences


def test_split_sentences_latin_terminators():
    text = "First sentence. Second sentence! Third one?"
    assert split_sentences(text) == ["First sentence.", "Second sentence!", "Third one?"]


def test_split_sentences_devanagari_danda():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("।")


def test_split_sentences_empty_text_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_no_terminator_returns_whole_text():
    assert split_sentences("no terminator here") == ["no terminator here"]


def test_sentence_spans_round_trip():
    text = "First sentence. Second sentence! Third one?"
    for sentence, (start, end) in sentence_spans(text):
        assert text[start:end] == sentence
