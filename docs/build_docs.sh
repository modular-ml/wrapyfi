#!/bin/bash
# uncomment following line, and rename index_API.rst to index.rst to get docs from code.
#sphinx-apidoc -o source ../wrapyfi
#rm -rf usage/
mdsplit usage.md -l 2 -o usage
sphinx-build -b html . _build

