#!/bin/bash

set -e
export PYTHONPATH=src

python3 -m unittest discover -s tests -p '*.py'
