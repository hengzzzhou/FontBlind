"""VLM evaluation runner — sends benchmark questions to models via API."""
import base64
import time
import openai
from fontbench.config import API_BASE_URL, API_KEY


def _encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMEvaluator:
    def __init__(self, model_id, model_name, max_retries=3):
        self.model_id = model_id
        self.model_name = model_name
        self.max_retries = max_retries
        self.client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def _call_api(self, messages):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"ERROR: {e}"

    def evaluate_mc(self, image_path, question, options):
        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        prompt = f"{question}\n\n{options_str}\n\nAnswer with only the letter (A, B, C, or D)."

        b64_image = _encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = self._call_api(messages)

        # Parse answer letter
        parsed = None
        for char in response.strip().upper():
            if char in "ABCD":
                idx = ord(char) - 65
                if idx < len(options):
                    parsed = options[idx]
                break

        return {
            "response": response,
            "parsed_answer": parsed,
            "options": options,
        }

    def evaluate_open_ended(self, image_path, question):
        b64_image = _encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": question},
                ],
            }
        ]

        response = self._call_api(messages)
        return {"response": response}

    def evaluate_mc_with_cot(self, image_path, question, options):
        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        prompt = (
            f"{question}\n\n{options_str}\n\n"
            "Think step by step. First examine the stroke style and letterforms. "
            "Then estimate the size relative to the image. "
            "Then identify the weight and style. "
            "Finally, give your answer as a single letter (A, B, C, or D)."
        )

        b64_image = _encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = self._call_api(messages)

        parsed = None
        # Look for last letter mentioned in response
        for char in reversed(response.strip().upper()):
            if char in "ABCD":
                idx = ord(char) - 65
                if idx < len(options):
                    parsed = options[idx]
                break

        return {
            "response": response,
            "parsed_answer": parsed,
            "options": options,
        }
