#!/bin/bash

sphinx-apidoc -o source ../wrapyfi
sphinx-build -b html . _build

