#!/bin/bash

# This file is designated by the latexmkrc config to be run
# in case the compilation fails

echo '' ; echo "Running $0 on latexmk failure" ; echo ''

# For debugging purposes, comment out the line below:

./latexmk-after-success-clean.sh

# latexmk 'fails' if the tiniest thing goes wrong -- including duplicate bib entries
# normally the 'failure' is really a success, so may as well clean up
