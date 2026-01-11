import os
import pathlib


def _cuda_include_dirs() -> list[str]:
  dirs = []
  cuda_include = os.getenv("CUDA_INCLUDE_PATH", "")
  if cuda_include:
    for path in cuda_include.split(":"):
      if path and pathlib.Path(path).exists() and path not in dirs:
        dirs.append(path)
  cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
  if cuda_home:
    path = (pathlib.Path(cuda_home) / "include").as_posix()
    if pathlib.Path(path).exists() and path not in dirs:
      dirs.append(path)
  conda_prefix = os.getenv("CONDA_PREFIX")
  if conda_prefix:
    path = (pathlib.Path(conda_prefix) / "targets" / "x86_64-linux" / "include").as_posix()
    if pathlib.Path(path).exists() and path not in dirs:
      dirs.append(path)
  return dirs


def patch_tinygrad_cuda_includes() -> bool:
  try:
    import tinygrad.runtime.support.compiler_cuda as compiler_cuda
  except Exception:
    return False
  if getattr(compiler_cuda, "_exo_cuda_includes_patched", False):
    return True

  original_init = compiler_cuda.CUDACompiler.__init__

  def _init(self, arch: str, cache_key: str = "cuda"):
    original_init(self, arch, cache_key)
    for path in _cuda_include_dirs():
      flag = f"-I{path}"
      if flag not in self.compile_options:
        self.compile_options.append(flag)

  compiler_cuda.CUDACompiler.__init__ = _init
  compiler_cuda._exo_cuda_includes_patched = True
  return True
