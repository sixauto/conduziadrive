#!/bin/bash

set -e

python3 -m venv ./venv
. venv/bin/activate
pip3 install -r requirements.txt
python3 car-planning.py
