import asyncio
import numpy as np
from typing import Optional

from .config import get_simulation_config
from .fault_injector import flip_bits_numpy


def throttle_seconds(byte_len: int, gbps: Optional[float]) -> float:
  if gbps is None or gbps <= 0:
    return 0.0
  # gbps here is gigabits per second; 1 Gb = 1e9 bits
  # bytes_per_second = gbps * 1e9 / 8
  # sleep = bytes / bytes_per_second = bytes * 8 / (gbps * 1e9)
  return float(byte_len) * 8.0 / (gbps * 1e9)


def apply_ber_to_bytes(data: bytes, ber: Optional[float], seed: Optional[int] = None) -> bytes:
  if not data or ber is None or ber <= 0.0:
    return data
  arr = np.frombuffer(data, dtype=np.uint8)
  flipped = flip_bits_numpy(arr, ber, seed)
  return flipped.astype(np.uint8).tobytes()


async def maybe_throttle(byte_len: int, gbps: Optional[float]) -> None:
  sleep_s = throttle_seconds(byte_len, gbps)
  if sleep_s > 0:
    await asyncio.sleep(sleep_s)


def active_config():
  cfg = get_simulation_config()
  return cfg if cfg and cfg.enable else None