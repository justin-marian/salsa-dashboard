#!/bin/bash

# Create a virtual environment if it doesn't exist
python3 -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
