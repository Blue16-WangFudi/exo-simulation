#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-}"
ENGINE="${2:-tinygrad}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "Usage: scripts/download_model.sh <model-id> [engine]"
  echo "Example: scripts/download_model.sh llama-3.2-1b tinygrad"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXO_HOME="${EXO_HOME:-$ROOT_DIR/.cache/exo}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda/exo-gpu/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}. Set PYTHON_BIN or install the conda env."
  exit 1
fi

if [[ -e "${EXO_HOME}" && ! -w "${EXO_HOME}" ]]; then
  echo "No write permission to ${EXO_HOME}."
  echo "Fix with: sudo chown -R $(id -u):$(id -g) ${EXO_HOME}"
  exit 1
fi

MODEL_ID="${MODEL_ID}" ENGINE="${ENGINE}" "${PYTHON_BIN}" - <<'PY'
import asyncio
import os
import sys

from exo.download.new_shard_download import download_shard
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem
from exo.inference.inference_engine import inference_engine_classes
from exo.models import build_full_shard

model_id = os.environ["MODEL_ID"]
engine = os.environ["ENGINE"]
engine_classname = inference_engine_classes.get(engine, engine)

shard = build_full_shard(model_id, engine_classname)
if shard is None:
    print(f"Unsupported model '{model_id}' for inference engine '{engine_classname}'.")
    sys.exit(2)

on_progress = AsyncCallbackSystem[str, tuple[object, RepoProgressEvent]]()

async def main() -> None:
    print(f"Downloading {model_id} with engine {engine_classname} ...")
    path, _ = await download_shard(shard, engine_classname, on_progress)
    print(f"Download complete: {path}")

asyncio.run(main())
PY
