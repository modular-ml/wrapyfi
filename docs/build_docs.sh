#!/bin/bash
# remove the built doc directories (except for usage)
rm -rf _build examples source
# compile API documentation.
sphinx-apidoc -o source ../wrapyfi
# compile example documentation
sphinx-apidoc -o examples ../examples
# WARNING: Do not remove usage, since it is manually edited for now, given mdsplit cannot relink references
#rm -rf usage/
#mdsplit usage.md -l 2 -o usage
sphinx-build -b html . _build

