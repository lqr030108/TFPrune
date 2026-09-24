#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_qwen.sh [retention_ratio]
Default: 0.222. The retained content-token fraction must be in (0, 1].
Environment: GPUS=0 TASKS=pope PRETRAINED=<model-or-path> OUTPUT_DIR=<path> LIMIT=<optional>
HELP
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if (( $# > 1 )); then usage >&2; exit 2; fi
ratio="${1:-0.222}"
"${PYTHON:-python}" - "$ratio" <<'PY'
import sys
try:
    value = float(sys.argv[1])
    valid = 0 < value <= 1
except ValueError:
    valid = False
if not valid:
    sys.exit('retention_ratio must be a number in (0, 1]')
PY
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export LMMS_EVAL_PLUGINS=tfprune
output_dir="${OUTPUT_DIR:-${project_root}/results/qwen_r${ratio}}"
mkdir -p "$output_dir"
command=("${PYTHON:-python}" -m lmms_eval
  --model qwen2_5_vl_tfprune
  --model_args "pretrained=${PRETRAINED:-Qwen/Qwen2.5-VL-7B-Instruct},target_retention_ratio=${ratio}"
  --tasks "${TASKS:-pope}" --batch_size 1 --seed 0,1234,1234,1234
  --output_path "$output_dir" --log_samples)
if [[ -n "${LIMIT:-}" ]]; then command+=(--limit "$LIMIT"); fi
"${command[@]}" 2>&1 | tee "$output_dir/run.log"
