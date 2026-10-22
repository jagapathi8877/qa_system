import pytest

import qa_engine
from qa_engine import QAEngine


def test_definition_query_uses_document_definition_and_related_points(monkeypatch):
    text = (
        "TF-IDF (Numerical Example) Step 3 Calculate TF-IDF. "
        "The TF-IDF score for cat is 0.029 in Document 1 and 0 in Document 2.\n\n"
        "TF-IDF (Term Frequency — Inverse Document Frequency) "
        "TF-IDF (Term Frequency-Inverse Document Frequency) is a powerful "
        "technique used to analyze the importance of words in a document corpus. "
        "TF-IDF plays a crucial role in text mining, information retrieval, and "
        "document classification. What is TF-IDF? TF-IDF is a numerical statistic "
        "that reflects the significance of a word within a document relative to "
        "a collection of documents, known as a corpus. The idea behind TF-IDF is "
        "to quantify the importance of a term in a document with respect to its "
        "frequency in the document and its rarity across multiple documents."
    )
    engine = QAEngine(text)

    def unexpected_model_call(question, contexts):
        raise AssertionError(
            f"Unexpected model call for {question!r} with {len(contexts)} contexts."
        )

    monkeypatch.setattr(qa_engine.qa_model, "answer_from", unexpected_model_call)
    answers = []
    for question in (
        "What is TF-IDF?",
        "what is tfidf",
        "WHAT IS TF-IDF!!!",
        "What is Term Frequency-Inverse Document Frequency?",
        "Define tfidf",
    ):
        result = engine.ask(question, retriever="tfidf")

        assert result["status"] == "answered"
        assert result["category"] == "Definition"
        assert result["answer"].startswith(
            "Based on the dataset: TF-IDF "
            "(Term Frequency-Inverse Document Frequency) is "
        )
        assert "numerical statistic" in result["answer"]
        assert "information retrieval" in result["answer"]
        assert "frequency" in result["answer"]
        assert "powerful technique" not in result["answer"]
        assert "\ufffd" not in result["answer"]
        assert result["span"] in result["source_passage"]
        assert result["passage_index"] == 2
        answers.append(result["answer"])

    assert len(set(answers)) == 1


def test_definition_route_does_not_treat_comparison_as_definition():
    assert qa_engine._definition_topic(
        "What is the difference between TF and IDF?"
    ) is None


@pytest.mark.parametrize(
    "question",
    (
        "Derive TF-IDF for this document.",
        "Give an example of TFIDF.",
        "What are the applications of TF-IDF?",
    ),
)
def test_definition_route_preserves_non_definition_intents(question):
    assert qa_engine._definition_topic(question) is None


def test_definition_route_does_not_answer_when_no_definition_is_present():
    engine = QAEngine(
        "TF-IDF retrieval uses weighted terms to rank relevant documents "
        "across a collection."
    )

    assert qa_engine._definition_topic("What is TF-IDF?") == "TF-IDF"
    assert qa_engine._definition_answer(
        engine, "What is TF-IDF?", "TF-IDF", 0.0, "tfidf"
    ) is None


def test_topic_in_section_contents_is_not_mistaken_for_definition():
    heading = (
        "UNIT IV: Basic statistical features of NLP TFIDF, Vectorization: "
        "Encoders and Decoders, Normalization, Advanced feature engineering "
        "and NLP algorithms Basics of Word2Vec 3 Contents TF-IDF. Why simple "
        "word counts are not always sufficient."
    )
    topic_span = qa_engine._find_topic_span(heading, "tfidf")

    assert topic_span is not None
    assert not qa_engine._has_direct_definition(heading, topic_span)


def test_definition_query_uses_expansion_for_acronym_and_ignores_examples(monkeypatch):
    engine = QAEngine(
        "For example: Document: AI is powerful. AI is useful. AI is everywhere.\n\n"
        "Artificial intelligence is the simulation of human intelligence by "
        "computer systems. It includes learning, reasoning, perception, and "
        "language understanding."
    )

    def unexpected_model_call(question, contexts):
        raise AssertionError(
            f"Unverified model answer requested for {question!r}."
        )

    monkeypatch.setattr(qa_engine.qa_model, "answer_from", unexpected_model_call)
    result = engine.ask("what is ai", retriever="tfidf")

    assert result["status"] == "answered"
    assert "simulation of human intelligence" in result["answer"]
    assert "AI is powerful" not in result["answer"]
    assert result["span"] in result["source_passage"]


def test_definition_route_recognizes_happens_when_explanations(monkeypatch):
    engine = QAEngine(
        "Overfitting happens when a model learns training data too closely "
        "and performs poorly on new examples."
    )

    def unexpected_model_call(question, contexts):
        raise AssertionError(
            f"Unexpected model call for {question!r} with {len(contexts)} contexts."
        )

    monkeypatch.setattr(qa_engine.qa_model, "answer_from", unexpected_model_call)
    result = engine.ask("What is overfitting?", retriever="tfidf")

    assert result["status"] == "answered"
    assert result["category"] == "Definition"
    assert "happens when" in result["answer"]


def test_definition_question_without_explicit_definition_does_not_guess(monkeypatch):
    engine = QAEngine(
        "The document uses AI is powerful as a classroom example but does "
        "not define artificial intelligence."
    )

    def unexpected_model_call(question, contexts):
        raise AssertionError(
            f"Unverified model answer requested for {question!r}."
        )

    monkeypatch.setattr(qa_engine.qa_model, "answer_from", unexpected_model_call)
    result = engine.ask("What is AI?", retriever="tfidf")

    assert result["status"] == "no_answer"
    assert "no clear definition" in result["answer"].lower()
