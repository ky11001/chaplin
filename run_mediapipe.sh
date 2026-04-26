#!/usr/bin/env bash
set -euo pipefail

uv run --with-requirements requirements.txt --python 3.12 \
  main.py \
  config_filename=./configs/LRS3_V_WER19.1.ini \
  detector=mediapipe \
  "$@"
