#!/bin/bash

# Delete pycache folders
find . -type d -name "__pycache__" -exec rm -r {} +
# Delete venvironment directory
rm -rf .venv
