"""Shared helpers for the build scripts: GPU arch detection + output paths.

Every builder writes its TRT engine to:
    ../models/<arch>/<engine_subdir>/<file>.trt

where <arch> is the sm_XX compute capability of the GPU you're building on
(TensorRT bakes the arch into the engine, so the arch you build on IS the
arch the engine runs on).
"""
import os
import subprocess


def detect_arch() -> str:
    """Compute capability of the current GPU as 'sm_XX' (e.g., 'sm_90').

    Reads nvidia-smi. Falls back to 'sm_90' if no GPU is visible (the build
    will fail anyway, but you get a clearer error from TensorRT than from us).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip().splitlines()
        if out:
            return f"sm_{out[0].strip().replace('.', '')}"
    except Exception:
        pass
    return "sm_90"


def repo_root() -> str:
    """Path to the optimized/tensorRT/ directory (parent of build/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arch_dir(arch: str = None) -> str:
    """Resolve the per-arch output dir for engines: ../models/<arch>/."""
    if arch is None:
        arch = detect_arch()
    d = os.path.join(repo_root(), "models", arch)
    os.makedirs(d, exist_ok=True)
    return d


def onnx_dir(engine_subdir: str = None) -> str:
    """ONNX output dir. ONNX is arch-independent — the same .onnx file can be
    compiled into a .trt engine for any TRT-supported GPU.

    Resolution order:
      1. $SA3_ONNX_DIR (override; used by automation)
      2. ../../../../stable-audio-3-optimized/onnx/  — the HF model repo when
         checked out as a sibling of stable-audio-3 (canonical layout: github
         repo = code, HF repo = ONNX + TRT artifacts)
      3. <github_repo>/optimized/tensorRT/onnx/      — local fallback when no
         HF clone is around (a working area; not git-tracked here)
    """
    override = os.environ.get("SA3_ONNX_DIR")
    if override:
        d = override
    else:
        # repo_root() = .../stable-audio-3/optimized/tensorRT
        # HF sibling  = .../stable-audio-3-optimized
        github_root = os.path.dirname(os.path.dirname(os.path.dirname(repo_root())))
        hf_sibling = os.path.join(github_root, "stable-audio-3-optimized", "onnx")
        if os.path.isdir(os.path.dirname(hf_sibling)):
            d = hf_sibling
        else:
            d = os.path.join(repo_root(), "onnx")
    if engine_subdir:
        d = os.path.join(d, engine_subdir)
    os.makedirs(d, exist_ok=True)
    return d
