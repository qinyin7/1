#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-full_48gb_optimized}"
CPU_THREADS="${CPU_THREADS:-14}"
DIN_WEIGHT="${DIN_WEIGHT:-0.6}"
PR3_WEIGHT="${PR3_WEIGHT:-0.4}"
RRF_K="${RRF_K:-60}"
MMR_LAMBDA="${MMR_LAMBDA:-0.9}"
TOWER_EXPERIMENT="${TOWER_EXPERIMENT:-feature_tower_id_dropout}"
DIN_VARIANT="${DIN_VARIANT:-}"

mkdir -p logs "artifacts/${PROFILE}/rank_mix/snapshots"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"

echo "[START] $(date '+%F %T') ${PROFILE} main run"
echo "[ENV] pwd=$PWD"
python - <<'PY'
import torch

print("[ENV] cuda_available", torch.cuda.is_available())
print("[ENV] device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
nvidia-smi || true

echo "[STEP] prepare profile"
python scripts/run_experiment.py --profile "${PROFILE}" --stage prepare

for panel in valid full_val; do
  echo "[STEP] recall ${panel} itemcf_main"
  python scripts/run_experiment.py \
    --profile "${PROFILE}" \
    --stage recall \
    --experiment-id itemcf_main \
    --panel "${panel}"

  echo "[STEP] recall ${panel} content_text_category"
  python scripts/run_experiment.py \
    --profile "${PROFILE}" \
    --stage recall \
    --experiment-id content_text_category \
    --panel "${panel}"

  echo "[STEP] recall ${panel} ${TOWER_EXPERIMENT}"
  python scripts/run_experiment.py \
    --profile "${PROFILE}" \
    --stage recall \
    --experiment-id "${TOWER_EXPERIMENT}" \
    --panel "${panel}"
done

echo "[STEP] rankmix full_val main params"
rankmix_args=(
  scripts/run_rank_mix.py
  --profile "${PROFILE}" \
  --panel full_val \
  --cpu-threads "${CPU_THREADS}" \
  --din-weight "${DIN_WEIGHT}" \
  --pr3-weight "${PR3_WEIGHT}" \
  --rrf-k "${RRF_K}" \
  --mmr-lambda "${MMR_LAMBDA}"
)
if [[ -n "${DIN_VARIANT}" ]]; then
  rankmix_args+=(--din-variant "${DIN_VARIANT}")
fi
python "${rankmix_args[@]}"

TAG="dw${DIN_WEIGHT}_pw${PR3_WEIGHT}_rrf${RRF_K}_mmr${MMR_LAMBDA}"
TAG="${TAG//./}"
cp "artifacts/${PROFILE}/rank_mix/summary_full_val.csv" \
  "artifacts/${PROFILE}/rank_mix/snapshots/summary_full_val_${TAG}.csv"
cp "artifacts/${PROFILE}/rank_mix/results_full_val.json" \
  "artifacts/${PROFILE}/rank_mix/snapshots/results_full_val_${TAG}.json"
cp "artifacts/${PROFILE}/rank_mix/daily_stability_full_val.csv" \
  "artifacts/${PROFILE}/rank_mix/snapshots/daily_stability_full_val_${TAG}.csv"

echo "[DONE] $(date '+%F %T') ${PROFILE} main run"
