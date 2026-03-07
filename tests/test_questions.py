# tests/test_questions.py
"""Tests for question generator."""
from fontbench.questions import generate_mc_questions, generate_open_ended_question

SAMPLE_METADATA = {
    "font_family": "Arial",
    "font_size": 24,
    "font_size_bucket": "medium",
    "font_color": "red",
    "font_style": "bold",
    "script": "latin",
    "sub_script": "latin",
}


def test_mc_question_per_property():
    questions = generate_mc_questions(SAMPLE_METADATA)
    assert len(questions) == 4  # family, size, style, color
    for q in questions:
        assert "question" in q
        assert "options" in q
        assert len(q["options"]) == 4
        assert "answer" in q
        assert q["answer"] in q["options"]
        assert "property" in q


def test_mc_question_font_family():
    questions = generate_mc_questions(SAMPLE_METADATA)
    family_q = [q for q in questions if q["property"] == "font_family"][0]
    assert "Arial" in family_q["options"]
    assert family_q["answer"] == "Arial"


def test_open_ended_question():
    q = generate_open_ended_question(SAMPLE_METADATA)
    assert "question" in q
    assert "ground_truth" in q
    assert q["ground_truth"]["font_family"] == "Arial"
    assert q["ground_truth"]["font_color"] == "red"
