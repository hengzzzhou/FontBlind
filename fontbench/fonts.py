"""Font registry — maps font names to system font files."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Font:
    name: str
    path: Path
    script: str  # "latin", "cjk", "other"
    sub_script: str  # "latin", "chinese", "arabic", "devanagari"
    styles: list = field(default_factory=lambda: ["regular"])


SYSTEM_FONT_DIRS = [
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
]

# Map font display names to (filename, script, sub_script, available_styles)
FONT_DEFINITIONS = [
    # Latin fonts
    ("Arial", "Arial.ttf", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Arial Bold", "Arial Bold.ttf", "latin", "latin", ["bold"]),
    ("Helvetica", "Helvetica.ttc", "latin", "latin", ["regular", "bold"]),
    ("Helvetica Neue", "HelveticaNeue.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Times New Roman", "Times New Roman.ttf", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Georgia", "Georgia.ttf", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Courier New", "Courier New.ttf", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Verdana", "Verdana.ttf", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Palatino", "Palatino.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Futura", "Futura.ttc", "latin", "latin", ["regular", "bold"]),
    ("Avenir", "Avenir.ttc", "latin", "latin", ["regular", "bold"]),
    ("Baskerville", "Baskerville.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Didot", "Didot.ttc", "latin", "latin", ["regular", "bold", "italic"]),
    ("Gill Sans", "GillSans.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Optima", "Optima.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Trebuchet MS", "Trebuchet MS.ttf", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("American Typewriter", "AmericanTypewriter.ttc", "latin", "latin", ["regular", "bold"]),
    ("Rockwell", "Rockwell.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Cochin", "Cochin.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    ("Menlo", "Menlo.ttc", "latin", "latin", ["regular", "bold", "italic", "bold_italic"]),
    # CJK fonts (Chinese only)
    ("STHeiti", "STHeiti Medium.ttc", "cjk", "chinese", ["regular", "bold"]),
    ("Songti SC", "Songti.ttc", "cjk", "chinese", ["regular", "bold"]),
    # Other scripts
    ("Al Nile", "Al Nile.ttc", "other", "arabic", ["regular", "bold"]),
    ("Baghdad", "Baghdad.ttc", "other", "arabic", ["regular", "bold"]),
    ("Devanagari MT", "DevanagariMT.ttc", "other", "devanagari", ["regular", "bold"]),
    ("Devanagari Sangam MN", "Devanagari Sangam MN.ttc", "other", "devanagari", ["regular", "bold"]),
]


def _find_font_path(filename):
    for font_dir in SYSTEM_FONT_DIRS:
        candidate = font_dir / filename
        if candidate.exists():
            return candidate
    return None


class FontRegistry:
    def __init__(self):
        self._fonts = []
        for name, filename, script, sub_script, styles in FONT_DEFINITIONS:
            path = _find_font_path(filename)
            if path is not None:
                self._fonts.append(Font(name=name, path=path, script=script, sub_script=sub_script, styles=styles))

    def get_fonts_by_script(self, script):
        return [f for f in self._fonts if f.script == script]

    def get_fonts_by_sub_script(self, sub_script):
        return [f for f in self._fonts if f.sub_script == sub_script]

    def all_fonts(self):
        return list(self._fonts)

    def get_font(self, name):
        for f in self._fonts:
            if f.name == name:
                return f
        return None
