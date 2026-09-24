#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_llava.sh [llava15|llava_next] [retained_tokens]
Defaults: llava15 128; llava_next 320.
Environment: GPUS=0 TASKS=pope PRETRAINED=<model-or-path> OUTPUT_DIR=<path> LIMIT=<optional>
HELP
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if (( $# > 2 )); then usage >&2; exit 2; fi
variant="${1:-llava15}"
case "$variant" in
  llava15) model="liuhaotian/llava-v1.5-7b"; budget="${2:-128}" ;;
  llava_next) model="liuhaotian/llava-v1.6-vicuna-7b"; budget="${2:-320}" ;;
  *) usage >&2; exit 2 ;;
esac
if [[ ! "$budget" =~ ^[1-9][0-9]*$ ]]; then
  echo 'retained_tokens must be a positive integer' >&2; exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export LMMS_EVAL_PLUGINS=tfprune
output_dir="${OUTPUT_DIR:-${project_root}/results/${variant}_k${budget}}"
mkdir -p "$output_dir"
command=("${PYTHON:-python}" -m lmms_eval
  --model llava_tfprune
  --model_args "pretrained=${PRETRAINED:-$model},target_vision_tokens=${budget}"
  --tasks "${TASKS:-pope}" --batch_size 1 --seed 0,1234,1234,1234
  --output_path "$output_dir" --log_samples)
if [[ -n "${LIMIT:-}" ]]; then command+=(--limit "$LIMIT"); fi
"${command[@]}" 2>&1 | tee "$output_dir/run.log"
