#!/bin/bash
# One command: install, download labels, embed, evaluate.
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python -m app.ingest
python -m app.index
python -m evals.run_eval
