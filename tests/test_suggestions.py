from pathlib import Path

import qa_engine
from qa_engine import QAEngine


SAMPLE_PATH = Path(__file__).parents[1] / "static" / "sample_dataset.txt"


def test_suggested_questions_return_their_source_text_without_qa_model(monkeypatch):
    engine = QAEngine(SAMPLE_PATH.read_text(encoding="utf-8"))
    questions = engine.suggest_questions()
    assert questions

    def unexpected_model_call():
        raise AssertionError("Suggested questions should use their verified source text.")

    monkeypatch.setattr(qa_engine.qa_model, "get_pipeline", unexpected_model_call)

    for question in questions:
        result = engine.ask(question.upper())

        assert result["status"] == "answered"
        # Source copying is verifiable; it is not a calibrated probability.
        assert result['confidence'] is None
        assert result['answer_method'] == 'suggested_source_evidence'
        assert result['confidence_calibrated'] is False
        assert result["span"] in result["source_passage"]
        assert result["span"] in result["answer"]
        assert result["answer_start"] is not None
        assert result["answer_end"] > result["answer_start"]
        assert "not enough to answer" not in result["answer"].lower()


def test_suggestions_scan_late_passages_and_prioritize_repeated_topics():
    filler = [
        (
            f"Background section {index} describes historical examples and "
            "provides additional context for the wider research topic."
        )
        for index in range(45)
    ]
    filler.extend(
        [
            "Quantum retrieval is a central method for finding relevant evidence "
            "across large scientific documents.",
            "Quantum retrieval prioritizes important evidence and improves "
            "search quality across scientific documents.",
            "Researchers emphasize quantum retrieval when evaluating evidence "
            "ranking for large documents.",
        ]
    )

    engine = QAEngine("\n\n".join(filler))
    questions = engine.suggest_questions()

    assert any("quantum retrieval" in question.lower() for question in questions)
    assert len(questions) == len(set(questions))


def test_suggestions_from_late_passages_return_their_source_without_model(monkeypatch):
    text = "\n\n".join(
        f"Background passage {index} includes broad context about the project."
        for index in range(45)
    )
    text += (
        "\n\nQuantum retrieval is a method for finding the strongest evidence "
        "in a scientific document."
    )
    engine = QAEngine(text)
    question = next(
        q for q in engine.suggest_questions(100)
        if "quantum retrieval" in q.lower()
    )

    def unexpected_model_call():
        raise AssertionError("Suggested questions should use their verified source text.")

    monkeypatch.setattr(qa_engine.qa_model, "get_pipeline", unexpected_model_call)
    result = engine.ask(question)

    assert result["status"] == "answered"
    assert result["span"] in result["source_passage"]
