# tests/test_scoring.py
"""Tests for scoring module."""
from fontbench.scoring import score_mc_results, score_open_ended_results


def test_score_mc_perfect():
    results = [
        {"property": "font_family", "answer": "Arial", "parsed_answer": "Arial"},
        {"property": "font_color", "answer": "red", "parsed_answer": "red"},
    ]
    scores = score_mc_results(results)
    assert scores["overall_accuracy"] == 1.0
    assert scores["per_property"]["font_family"] == 1.0
    assert scores["per_property"]["font_color"] == 1.0


def test_score_mc_partial():
    results = [
        {"property": "font_family", "answer": "Arial", "parsed_answer": "Georgia"},
        {"property": "font_family", "answer": "Helvetica", "parsed_answer": "Helvetica"},
        {"property": "font_color", "answer": "red", "parsed_answer": "red"},
    ]
    scores = score_mc_results(results)
    assert scores["overall_accuracy"] == 2 / 3
    assert scores["per_property"]["font_family"] == 0.5


def test_score_mc_with_none():
    results = [
        {"property": "font_family", "answer": "Arial", "parsed_answer": None},
    ]
    scores = score_mc_results(results)
    assert scores["overall_accuracy"] == 0.0


def test_score_mc_by_dimension():
    results = [
        {"property": "font_family", "answer": "Arial", "parsed_answer": "Arial",
         "difficulty": "easy", "script": "latin", "source": "synthetic"},
        {"property": "font_family", "answer": "STHeiti", "parsed_answer": "Songti SC",
         "difficulty": "hard", "script": "cjk", "source": "synthetic"},
    ]
    scores = score_mc_results(results)
    assert scores["by_difficulty"]["easy"] == 1.0
    assert scores["by_difficulty"]["hard"] == 0.0
    assert scores["by_script"]["latin"] == 1.0
    assert scores["by_script"]["cjk"] == 0.0
