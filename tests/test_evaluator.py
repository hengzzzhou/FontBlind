# tests/test_evaluator.py
"""Tests for VLM evaluator."""
from unittest.mock import patch, MagicMock
from fontbench.evaluator import VLMEvaluator


def _mock_chat_response(content):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    return mock_resp


def test_evaluator_init():
    evaluator = VLMEvaluator(model_id="test-model", model_name="Test")
    assert evaluator.model_id == "test-model"


@patch("fontbench.evaluator._encode_image", return_value="fakebase64data")
@patch("fontbench.evaluator.openai.OpenAI")
def test_evaluate_mc_question(mock_openai_cls, mock_encode):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("A")

    evaluator = VLMEvaluator(model_id="test-model", model_name="Test")
    result = evaluator.evaluate_mc(
        image_path="/fake/path.png",
        question="What font is this?",
        options=["Arial", "Georgia", "Verdana", "Courier"],
    )
    assert "response" in result
    assert "parsed_answer" in result
    assert result["parsed_answer"] == "Arial"


@patch("fontbench.evaluator._encode_image", return_value="fakebase64data")
@patch("fontbench.evaluator.openai.OpenAI")
def test_evaluate_open_ended(mock_openai_cls, mock_encode):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response(
        "The font is Arial, medium size, bold, black color."
    )

    evaluator = VLMEvaluator(model_id="test-model", model_name="Test")
    result = evaluator.evaluate_open_ended(
        image_path="/fake/path.png",
        question="Describe the font properties.",
    )
    assert "response" in result
