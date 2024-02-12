from sphinx.application import Sphinx
import re


def convert_math_blocks(app: Sphinx, docname: str, source: list):
    if source:
        source[0] = re.sub(r"```math", "```{math}", source[0])


def setup(app: Sphinx):
    app.connect("source-read", convert_math_blocks)
