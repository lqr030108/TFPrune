# TFPrune

TFPrune supports visual-token compression for LLaVA and Qwen2.5-VL. We evaluate the models using **lmms-eval**. The two scripts below launch lmms-eval with the TFPrune model adapters. Follow the official installation guide to install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Run all commands from the extracted `TFPrune` directory. Linux, Git, a CUDA-capable GPU, and Python 3.10 are recommended.

## LLaVA

**Environment Setup**

```bash
conda create -n tfprune-llava python=3.10 -y
conda activate tfprune-llava
python -m pip install -r requirements/llava.txt
python -m pip install -e . --no-deps
```

Install LLaVA-NeXT

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]" --no-deps
cd ..
```

**Run LLaVA-1.5-7B**

The final argument is the number of visual content tokens to retain:

```bash
GPUS=0 TASKS=pope bash scripts/run_llava.sh llava15 128
```

Other budgets or the Vicuna-based LLaVA-NeXT checkpoint:

```bash
GPUS=0 TASKS=pope bash scripts/run_llava.sh llava15 64
GPUS=0 TASKS=pope bash scripts/run_llava.sh llava_next 320
```

The default checkpoints are `liuhaotian/llava-v1.5-7b` and `liuhaotian/llava-v1.6-vicuna-7b`.

## Qwen2.5-VL

**Environment Setup**

Use a separate environment because Qwen and LLaVA require different Transformers versions:

```bash
conda create -n tfprune-qwen python=3.10 -y
conda activate tfprune-qwen
python -m pip install -r requirements/qwen.txt
python -m pip install -e . --no-deps
```

The final argument is the fraction of visual content tokens to retain, between 0 and 1:

```bash
GPUS=0 TASKS=pope bash scripts/run_qwen.sh 0.222
```

The default checkpoint is `Qwen/Qwen2.5-VL-7B-Instruct`. For example, `0.222` retains approximately 22.2% of the image tokens; `1.0` keeps the full visual budget.

## Common options

Set `GPUS` to the visible GPU IDs, `TASKS` to an lmms-eval task name or comma-separated list, and optionally `OUTPUT_DIR` to choose where results are written. Use `LIMIT` for a quick installation check:

```bash
GPUS=0 TASKS=pope LIMIT=8 OUTPUT_DIR=results/quick_check \
  bash scripts/run_llava.sh llava15 128
```

Without `LIMIT`, the selected task runs in full. Results and `run.log` are written under `results/` by default. Run one process with batch size 1.
