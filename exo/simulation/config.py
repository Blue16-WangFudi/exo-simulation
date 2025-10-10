import os
import json
from typing import Optional, Any, Dict
from pydantic import BaseModel, ValidationError


DEFAULT_CONFIG_PATH = os.getenv("EXO_SIMULATION_CONFIG", "exo/config/simulation.json")
_CACHED_CONFIG: Optional["SimulationConfig"] = None


class SimulationConfig(BaseModel):
  enable: bool = False
  seed: Optional[int] = None
  tops_override_fp16_tflops: Optional[float] = None
  tops_scale: Optional[float] = None
  ber: Optional[float] = None
  ber_scope: Optional[str] = None  # one of: forward_tensor, output_tensor, token_sampling
  weighting_alpha: Optional[float] = None


def _read_profile_name_from_file() -> Optional[str]:
  """Read active profile name from file if present.

  Path can be overridden with EXO_SIMULATION_PROFILE_FILE env; defaults to
  "exo/config/.simulation_profile".
  """
  profile_file = os.getenv("EXO_SIMULATION_PROFILE_FILE", "exo/config/.simulation_profile")
  try:
    with open(profile_file, "r") as f:
      content = f.read().strip()
      return content or None
  except FileNotFoundError:
    return None
  except Exception:
    return None


def _select_active_profile(root_obj: Dict[str, Any]) -> Optional[SimulationConfig]:
  """Support multi-profile configs.

  Expected structure:
  {
    "profiles": { "dev": { ... }, "gpu": { ... } },
    "default_profile": "dev"  # optional
  }

  Falls back to env EXO_SIMULATION_PROFILE or file selector.
  """
  profiles = root_obj.get("profiles")
  if not isinstance(profiles, dict):
    return None

  # Resolve desired profile: env -> file -> json key -> first available
  profile_from_env = os.getenv("EXO_SIMULATION_PROFILE")
  profile_from_file = _read_profile_name_from_file()
  default_profile = root_obj.get("default_profile")

  active_name = profile_from_env or profile_from_file or default_profile
  if not active_name:
    # Pick first entry deterministically
    try:
      active_name = next(iter(profiles.keys()))
    except StopIteration:
      return None

  profile_data = profiles.get(active_name)
  if not isinstance(profile_data, dict):
    # If specified name is missing, try default or first
    if default_profile and isinstance(profiles.get(default_profile), dict):
      profile_data = profiles[default_profile]
    else:
      # First valid dict profile
      for _, v in profiles.items():
        if isinstance(v, dict):
          profile_data = v
          break
  if not isinstance(profile_data, dict):
    return None

  try:
    return SimulationConfig.model_validate(profile_data)
  except ValidationError:
    return None


def _read_config(path: str) -> Optional[SimulationConfig]:
  try:
    with open(path, "r") as f:
      raw = f.read()
  except FileNotFoundError:
    return None

  # Try multi-profile format first
  try:
    obj = json.loads(raw)
  except Exception:
    obj = None

  if isinstance(obj, dict):
    selected = _select_active_profile(obj)
    if selected is not None:
      return selected

  # Fallback to single-profile format (backward compatible)
  try:
    return SimulationConfig.model_validate_json(raw)
  except ValidationError:
    return None


def load_global_config() -> Optional[SimulationConfig]:
  return _read_config(DEFAULT_CONFIG_PATH)


def get_simulation_config() -> Optional[SimulationConfig]:
  global _CACHED_CONFIG
  if _CACHED_CONFIG is not None:
    return _CACHED_CONFIG
  _CACHED_CONFIG = load_global_config()
  return _CACHED_CONFIG