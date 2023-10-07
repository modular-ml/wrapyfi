#!/bin/bash
# uncomment following line, and rename index_API.rst to index.rst to get docs from code.
#sphinx-apidoc -o source ../wrapyfi
# uncomment following lines to compile example documentation
python build_examples.py
# WARNING: Do not remove usage, since it is manually edited for now, given mdsplit cannot relink references
#rm -rf usage/
mdsplit usage.md -l 2 -o usage
sphinx-build -b html . _build

