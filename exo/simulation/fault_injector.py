import numpy as np
from typing import Optional


def _to_uint_view(arr: np.ndarray) -> tuple[np.ndarray, int]:
  """Return a uint view and bits per element, supporting float and int arrays."""
  kind = arr.dtype.kind
  itemsize = arr.dtype.itemsize
  bits = itemsize * 8
  if kind == 'f':
    if itemsize == 2:
      return arr.view(np.uint16), bits
    elif itemsize == 4:
      return arr.view(np.uint32), bits
    elif itemsize == 8:
      return arr.view(np.uint64), bits
  elif kind == 'i':
    if itemsize == 1:
      return arr.view(np.uint8), bits
    elif itemsize == 2:
      return arr.view(np.uint16), bits
    elif itemsize == 4:
      return arr.view(np.uint32), bits
    elif itemsize == 8:
      return arr.view(np.uint64), bits
  elif kind == 'u':
    if itemsize == 1:
      return arr.view(np.uint8), bits
    elif itemsize == 2:
      return arr.view(np.uint16), bits
    elif itemsize == 4:
      return arr.view(np.uint32), bits
    elif itemsize == 8:
      return arr.view(np.uint64), bits
  # Unsupported kinds: return original as uint8 fallback
  return arr.view(np.uint8), 8


def flip_bits_numpy(arr: np.ndarray, ber: float, seed: Optional[int] = None) -> np.ndarray:
  """Flip bits in a numpy array according to BER.

  - ber: probability for each bit to flip independently.
  - seed: optional seed for reproducibility.

  Returns a new array with the same dtype and shape.
  """
  if arr.size == 0 or ber is None or ber <= 0.0:
    return arr
  rng = np.random.default_rng(seed)

  # Work on a flattened view to simplify mask creation
  original_dtype = arr.dtype
  flat = arr.reshape(-1)
  uint_view, bits_per_elem = _to_uint_view(flat)

  # Create bit masks per element: each bit flips with probability ber
  # Shape: (n, bits_per_elem) booleans
  n = uint_view.size
  prob = rng.random((n, bits_per_elem)) < ber
  # Convert boolean bit positions into integer masks
  # weights: [1, 2, 4, ...]
  weights = (1 << np.arange(bits_per_elem, dtype=uint_view.dtype))
  mask = (prob * weights).sum(axis=1).astype(uint_view.dtype)

  # Apply XOR flip
  flipped = (uint_view ^ mask).view(original_dtype)
  return flipped.reshape(arr.shape)


def inject_token_error(tokens: np.ndarray, ber: float, seed: Optional[int] = None) -> np.ndarray:
  """Inject token-level errors by random +/-1 perturbations with probability ber per token.
  Assumes integer dtype tokens.
  """
  if tokens.size == 0 or ber is None or ber <= 0.0:
    return tokens
  if tokens.dtype.kind not in ('i', 'u'):
    return tokens

  rng = np.random.default_rng(seed)
  perturbed = tokens.copy()
  flip_mask = rng.random(size=perturbed.shape) < ber
  # +/-1 change
  delta = rng.choice([-1, 1], size=perturbed.shape)
  perturbed = np.where(flip_mask, perturbed + delta, perturbed)
  return perturbed