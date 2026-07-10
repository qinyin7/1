#!/usr/bin/env bash
set -euo pipefail

export PROFILE="${PROFILE:-full_48gb_boosted}"
export TOWER_EXPERIMENT="${TOWER_EXPERIMENT:-feature_tower_dropout_hard_negative}"
export CPU_THREADS="${CPU_THREADS:-14}"
export DIN_WEIGHT="${DIN_WEIGHT:-0.7}"
export PR3_WEIGHT="${PR3_WEIGHT:-0.3}"
export RRF_K="${RRF_K:-60}"
export MMR_LAMBDA="${MMR_LAMBDA:-0.9}"
export DIN_VARIANT="${DIN_VARIANT:-multibehavior}"

bash scripts/run_full_48gb_main.sh
