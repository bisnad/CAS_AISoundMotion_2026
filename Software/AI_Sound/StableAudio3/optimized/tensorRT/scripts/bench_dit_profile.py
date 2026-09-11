"""DiT-only timing benchmark across multiple engines + L values.

Loads each engine once, runs dit.step() in a tight loop at every requested L
(skipping L below the engine's profile min), captures fp32 timings with
torch.cuda.Event, and reports median across N runs.

Usage:
    python bench_dit_profile.py \
        --engines '<label>=<path>,<label>=<path>,...' \
        --lvals 32,64,128,256,512,1024,2048,4096 \
        --warmup 3 --runs 5 [--seed 0]

Engines run sequentially; each is loaded, benchmarked at all valid L values,
then freed before the next engine. Results are emitted as a TSV table to
stdout (engine, L, median_ms, mean_ms, min_ms, max_ms, runs).
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


# ── lazy heavy imports (torch + trt + plugin) ─────────────────────────────
@contextlib.contextmanager
def _silence_fd(fileno: int):
    saved = os.dup(fileno)
    try:
        with open(os.devnull, "wb") as null:
            os.dup2(null.fileno(), fileno)
        yield
    finally:
        os.dup2(saved, fileno)
        os.close(saved)


def _import_heavy():
    """Mirror sa3_trt._import_heavy() so its module globals (torch, trt) are
    populated — TRTRunner/DiTRunner reach into sa3_trt's globals.
    """
    global torch, trt
    import torch as _torch
    import tensorrt as _trt
    torch = _torch
    trt = _trt
    # Inject torch/trt into sa3_trt_core's globals (formerly sa3_trt before the
    # full-pipeline-graph refactor); its classes TRTRunner / DiTRunner reach
    # into the module globals to find these.
    import sa3_trt_core as _s
    _s.torch = _torch
    _s.trt = _trt
    with _silence_fd(1), _silence_fd(2):
        try:
            import diff_attn_nocast_plugin  # noqa: F401
        except Exception:
            pass


def _engine_profile_min_L(engine, ctx) -> int:
    """Read input-shape lower bound on 'x' from the engine's optimization profile.

    TRT exposes profile shapes via Engine.get_tensor_profile_shape(name, profile_idx).
    Returns the min L on the last (sequence) axis of 'x'.
    """
    lo, opt, hi = engine.get_tensor_profile_shape("x", 0)
    return int(lo[-1])


def _engine_profile_max_L(engine) -> int:
    lo, opt, hi = engine.get_tensor_profile_shape("x", 0)
    return int(hi[-1])


def bench_one_engine(engine_path: str, label: str, lvals: list[int],
                     warmup: int, runs: int, seed: int) -> list[dict]:
    """Load `engine_path`, benchmark dit.step() at each L in `lvals` that fits
    within the engine's profile. Returns one dict per (L) measurement.
    """
    from sa3_trt_core import TRTRunner, DiTRunner, IO_CHANNELS, T5_MAX_LEN, COND_DIM

    print(f"\n━━━ engine: {label} ━━━", flush=True)
    print(f"  path: {engine_path}", flush=True)
    sz = Path(engine_path).stat().st_size
    print(f"  size: {sz/1e9:.2f} GB", flush=True)

    runner = TRTRunner(Path(engine_path))
    dit = DiTRunner(runner)

    lo = _engine_profile_min_L(runner.engine, runner.context)
    hi = _engine_profile_max_L(runner.engine)
    print(f"  profile min/max L: {lo}/{hi}", flush=True)

    torch.manual_seed(seed)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    rows = []
    valid_lvals = [L for L in lvals if lo <= L <= hi]
    skipped = [L for L in lvals if L < lo or L > hi]
    if skipped:
        print(f"  skipping L outside profile: {skipped}", flush=True)

    for L in valid_lvals:
        # Inputs sized for this L (fp32 throughout; DiTRunner casts internally
        # in step()).
        x = torch.randn(1, IO_CHANNELS, L, device="cuda", dtype=torch.float32, generator=g)
        t_val = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        t5_h = torch.randn(1, T5_MAX_LEN, COND_DIM, device="cuda", dtype=torch.float32, generator=g)
        t5_m = torch.ones(1, T5_MAX_LEN, device="cuda", dtype=torch.float32)
        sec  = torch.tensor([float(L * 4096 / 44100)], device="cuda", dtype=torch.float32)
        lac  = torch.zeros(1, 257, L, device="cuda", dtype=torch.float32)

        # Warmup (also forces _setup at this L, which sets input shape on the ctx).
        for _ in range(warmup):
            _ = dit.step(x, t_val, t5_h, t5_m, sec, lac)
        torch.cuda.synchronize()

        # Timed runs (per-step timing — the engine's stream is sync'd inside
        # step() already, so each iteration is one DiT forward end-to-end).
        times_ms = []
        for _ in range(runs):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            _ = dit.step(x, t_val, t5_h, t5_m, sec, lac)
            ev1.record()
            torch.cuda.synchronize()
            times_ms.append(ev0.elapsed_time(ev1))

        med = float(np.median(times_ms))
        mn  = float(np.mean(times_ms))
        lo_t = float(min(times_ms))
        hi_t = float(max(times_ms))
        print(f"    L={L:>5}  median={med:7.2f} ms  mean={mn:7.2f}  min={lo_t:7.2f}  max={hi_t:7.2f}  "
              f"(n={runs})", flush=True)
        rows.append({
            "engine": label,
            "engine_path": engine_path,
            "engine_size_bytes": sz,
            "profile_min_L": lo,
            "profile_max_L": hi,
            "L": L,
            "median_ms": med,
            "mean_ms": mn,
            "min_ms": lo_t,
            "max_ms": hi_t,
            "runs": runs,
        })

        # Free the inputs before the next L (they grow with L for x/lac).
        del x, t5_h, t5_m, t_val, sec, lac
        torch.cuda.empty_cache()

    # Free the engine + context before loading the next one.
    runner.free()
    del runner, dit
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", required=True,
                    help="Comma-sep list of label=path entries, e.g. "
                         "'canonical=/.../dit_bf16.trt,min32=/.../dit_bf16_min32.trt'")
    ap.add_argument("--lvals", default="32,64,128,256,512,1024,2048,4096",
                    help="Comma-sep L values to benchmark at. Engines skip L < their profile min.")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs",   type=int, default=5)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--tsv",    default=None,
                    help="Optional path to write rows as TSV (in addition to stdout).")
    args = ap.parse_args()

    _import_heavy()

    engines = []
    for chunk in args.engines.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            sys.exit(f"--engines entry must be 'label=path', got {chunk!r}")
        label, path = chunk.split("=", 1)
        engines.append((label.strip(), path.strip()))

    lvals = [int(x) for x in args.lvals.split(",") if x.strip()]

    all_rows = []
    for label, path in engines:
        rows = bench_one_engine(path, label, lvals, args.warmup, args.runs, args.seed)
        all_rows.extend(rows)

    # Final summary table — pivot by L → engine.
    print("\n━━━ summary (median ms, n={}) ━━━".format(args.runs))
    eng_labels = [e[0] for e in engines]
    header = ["L"] + eng_labels
    print("\t".join(header))
    by_eng_L = {(r["engine"], r["L"]): r for r in all_rows}
    for L in lvals:
        row = [str(L)]
        for el in eng_labels:
            r = by_eng_L.get((el, L))
            row.append(f"{r['median_ms']:.2f}" if r else "—")
        print("\t".join(row))

    if args.tsv:
        import csv
        keys = ["engine", "engine_path", "engine_size_bytes", "profile_min_L",
                "profile_max_L", "L", "median_ms", "mean_ms", "min_ms", "max_ms", "runs"]
        with open(args.tsv, "w") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nwrote {args.tsv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
