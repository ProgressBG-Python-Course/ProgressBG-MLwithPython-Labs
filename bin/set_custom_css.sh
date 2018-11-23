#!/bin/bash

# echo $PWD

CUSTOM_CSS_FILE="$PWD/bin/custom.css"
JUPY_CSS_FILE="$PWD/.venv/lib/python3.7/site-packages/notebook/static/custom/custom.css"
if [ -f $JUPY_CSS_FILE ]; then
  cp $CUSTOM_CSS_FILE $JUPY_CSS_FILE
else
   echo "File $JUPY_CSS_FILE does not exist."
fi