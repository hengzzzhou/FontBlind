"""Prompt engineering strategies for font recognition."""


class ZeroShotStrategy:
    name = "zero_shot"

    def format_mc(self, question, options):
        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{question}\n\n{options_str}\n\nAnswer with only the letter (A, B, C, or D)."

    def format_open_ended(self, question):
        return question


class FewShotStrategy:
    name = "few_shot"

    FEW_SHOT_EXAMPLES = (
        "Example 1: The text in the image uses the Arial font family, medium size, "
        "regular style, and black color.\n"
        "Example 2: The text uses SimHei font, large size, bold style, red color.\n"
        "Example 3: The text uses Times New Roman, small size, italic style, blue color.\n\n"
    )

    def format_mc(self, question, options):
        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return (
            f"Here are some examples of font property identification:\n"
            f"{self.FEW_SHOT_EXAMPLES}"
            f"Now answer the following question.\n\n"
            f"{question}\n\n{options_str}\n\n"
            f"Answer with only the letter (A, B, C, or D)."
        )

    def format_open_ended(self, question):
        return (
            f"Here are some examples of font property identification:\n"
            f"{self.FEW_SHOT_EXAMPLES}"
            f"Now answer the following question.\n\n{question}"
        )


class CoTStrategy:
    name = "cot"

    def format_mc(self, question, options):
        options_str = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return (
            f"{question}\n\n{options_str}\n\n"
            f"Think step by step:\n"
            f"1. First, examine the stroke style and letterforms carefully.\n"
            f"2. Then, estimate the text size relative to the image dimensions.\n"
            f"3. Next, identify the weight (regular/bold) and style (upright/italic).\n"
            f"4. Finally, determine the color of the text.\n\n"
            f"After your analysis, give your final answer as a single letter (A, B, C, or D)."
        )

    def format_open_ended(self, question):
        return (
            f"{question}\n\n"
            f"Think step by step:\n"
            f"1. First, examine the stroke style and letterforms to identify the font family.\n"
            f"2. Then, estimate the text size (small, medium, large, or extra-large).\n"
            f"3. Next, identify the weight (regular/bold) and style (upright/italic).\n"
            f"4. Finally, determine the color of the text.\n\n"
            f"Provide your answer with all four properties: font family, size, style, and color."
        )


STRATEGIES = {
    "zero_shot": ZeroShotStrategy,
    "few_shot": FewShotStrategy,
    "cot": CoTStrategy,
}
