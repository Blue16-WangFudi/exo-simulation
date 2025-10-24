import os
import asyncio
from typing import Optional

from exo.simulation.config import get_simulation_config
from exo.inference.shard import Shard

# Reference TFLOPS to normalize override values into a [0, +inf) speed multiplier.
# You can tune this via env: EXO_REF_FP16_TFLOPS
DEFAULT_REF_FP16_TFLOPS = 100.0

# Base time per layer per token in milliseconds. Tunable via env: EXO_BASE_MS_PER_LAYER
DEFAULT_BASE_MS_PER_LAYER = 0.10  # 0.10ms per layer per token (approximation)


def get_throttle_multiplier() -> float:
  """Return speed multiplier in [~0, +inf).
  1.0 means no throttle. <1 slows down; >1 speeds up (negative delay clamped).

  Combines tops_override_fp16_tflops (as absolute, normalized to reference)
  and tops_scale (as relative multiplier). If simulation not enabled, returns 1.0.
  """
  sim = get_simulation_config()
  if not sim or not sim.enable:
    return 1.0

  # Start with neutral speed
  speed = 1.0

  # If override is present, normalize to reference TFLOPS to get a speed scale.
  if sim.tops_override_fp16_tflops is not None:
    try:
      ref = float(os.getenv("EXO_REF_FP16_TFLOPS", DEFAULT_REF_FP16_TFLOPS))
      override = max(0.0, float(sim.tops_override_fp16_tflops))
      # Avoid zero which would imply infinite delay; clamp to a small minimum
      speed = max(1e-4, override / ref)
    except Exception:
      pass

  # Apply relative scaling
  if sim.tops_scale is not None:
    try:
      speed *= float(sim.tops_scale)
    except Exception:
      pass

  # Clamp to sensible bounds
  speed = max(1e-4, min(speed, 100.0))
  return speed


async def throttle_sleep(shard: Optional[Shard], tokens_count: int) -> None:
  """Asynchronously sleep to simulate throughput constraints.

  - Delay is proportional to `n_layers * tokens_count`.
  - The delay scales inversely with the computed speed multiplier.
  - If speed >= 1, delay is clamped to zero (no wait).
  """
  try:
    base_ms_per_layer = float(os.getenv("EXO_BASE_MS_PER_LAYER", DEFAULT_BASE_MS_PER_LAYER))
    n_layers = shard.n_layers if shard is not None else 32
    speed = get_throttle_multiplier()

    # Compute delay: base time per layer per token multiplied by work units
    # and adjusted by speed multiplier (inverse).
    base_ms = base_ms_per_layer * max(1, n_layers) * max(1, tokens_count)
    # If speed==1.0 -> delay_factor = 0, speed<1 -> positive delay, speed>1 -> clamp to 0
    delay_ms = base_ms * (1.0 / speed - 1.0)
    if delay_ms <= 0:
      return

    await asyncio.sleep(delay_ms / 1000.0)
  except Exception:
    # Fail-open: never block if anything goes wrong
    return
