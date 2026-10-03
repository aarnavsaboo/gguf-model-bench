# gguf-model-bench

Local GGUF inspection and llama.cpp benchmark orchestration.

This repository combines two pieces that are useful when working with several local GGUF model variants:

1. inspect model metadata directly from the GGUF container;
2. run repeatable prompt-processing and text-generation benchmark matrices with llama.cpp.

The aim is to keep model-file metadata and runtime measurements in the same experiment workspace.

## What it does

- reads GGUF header and metadata without loading model tensors
- extracts architecture/model metadata when present
- records file size, tensor count and metadata count
- discovers GGUF files recursively
- groups variants by a configurable family label
- generates llama-bench command matrices
- runs prompt-processing and generation sweeps
- stores llama-bench JSON output without flattening away backend details
- compares prompt-processing throughput and generation throughput separately
- supports depth, batch-size and cache-type sweeps
- produces per-model summary tables

## Inspect models

```bash
python -m gguf_model_bench inspect models/ > runs/models.jsonl
```

A metadata row can include fields such as architecture, model name, parameter-related metadata and tokenizer information when those values exist in the file.

## Build a benchmark matrix

```bash
python -m gguf_model_bench plan configs/bench.example.json > runs/plan.jsonl
```

Then execute it:

```bash
python -m gguf_model_bench run runs/plan.jsonl --out runs/raw.jsonl
python -m gguf_model_bench report runs/raw.jsonl
```

## Benchmark dimensions

The planner can vary:

- GGUF file
- prompt tokens
- generated tokens
- context depth
- batch size
- micro-batch size
- KV cache type
- thread count
- GPU layer setting
- repetition count

The wrapper asks llama-bench for JSON output and stores both the command and parsed rows.

## Why separate prompt processing from generation?

Local inference workloads can bottleneck in different places. A RAG request with a large evidence block and a short answer is mostly a prompt-processing workload. A coding or writing task with a short prompt and long output can be generation-heavy.

Comparing both rates prevents a model/runtime configuration from looking universally fast based on one workload shape.

## Architecture

```text
GGUF files --------------------+
   |                           |
   v                           |
metadata reader                |
   |                           |
   +-----------> model catalog |
                               |
benchmark manifest ------------+
   |
   v
matrix planner
   |
   v
llama-bench runner
   |
   +--> command record
   +--> build/backend metadata
   +--> pp tokens/sec
   +--> tg tokens/sec
   +--> repetition statistics
   |
   v
raw JSONL -> report / comparisons
```

## Repository layout

- `gguf.py` — lightweight GGUF metadata reader
- `catalog.py` — recursive model discovery
- `planner.py` — benchmark matrix expansion
- `llama_bench.py` — llama-bench command and execution wrapper
- `report.py` — grouped throughput summaries
- `io.py` — JSONL helpers
- `configs/` — benchmark manifests
- `docs/` — file-format and methodology notes
- `tests/` — parser and planner tests

No model files are committed.

Maintained by **Aarnav Saboo**.
