#!/usr/bin/env python3
"""
ab_perf_test.py — A/B performance + peak-memory sweep for gpt-server-27b.

Measures, per (kv_bits x prefill_step x context_length) cell:
  - prefill throughput (tokens/s)
  - TTFT (end-to-end: prefill + cache copy + first decoded token)
  - decode throughput (tokens/s)
  - PEAK process memory (MLX peak + RSS) — the number that decides whether a
    config fits under your RAM limit.

It mirrors the server's memory shape: it builds the SAME quantized + recurrent
cache (_make_cache), takes a working COPY (_copy_cache) before decoding, and
decodes on the copy — so the measured peak includes the snapshot+copy 2x
transient that actually sets the high-water mark in production, on top of the
~17.5GB of resident weights.

This characterizes RAW MODEL performance. It does NOT exercise the server's
snapshot/disk/front-snapshot REUSE (that's loop logic — read it off the live
server's `kv-reuse%` / `warm-loaded` log lines instead).

RUN (on the Apple-Silicon box, same model as the server)
  export QWEN_MLX="mlx-community/Qwen3.6-27B-OptiQ-4bit"
  python ab_perf_test.py
  # narrow the sweep:
  python ab_perf_test.py --contexts 4096,32768,65536 --kv-bits 8,4 \
                         --prefill-steps 2048,4096 --decode 64 --mem-limit 40
  # add the 128k cell (slow — minutes of prefill):
  python ab_perf_test.py --contexts 4096,16384,32768,65536,131072

Results print as a table per config and are written to ./ab_results/<ts>.{json,csv}.
"""

import os
import sys
import csv
import json
import time
import argparse
import itertools

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None

# Base text tiled to hit exact context lengths. Code-ish so token stats are
# realistic; perf depends on token COUNT, not content, so this is plenty.
_BASE = (
    "def process(items):\n"
    "    results = []\n"
    "    for it in items:\n"
    "        if it.valid and it.score > threshold:\n"
    "            results.append(transform(it, config))\n"
    "    return aggregate(results)\n\n"
    "# This module handles the ingestion pipeline, validation, scoring, and\n"
    "# aggregation of incoming records from the upstream service queue.\n"
)


# --- MLX memory helpers (version-tolerant, mirrors the server) ------------- #
def _mx(name):
    for obj in (mx, getattr(mx, "metal", None)):
        if obj is None:
            continue
        fn = getattr(obj, name, None)
        if fn is not None:
            try:
                return fn()
            except Exception:
                return None
    return None


def _reset_peak():
    for obj in (mx, getattr(mx, "metal", None)):
        if obj is None:
            continue
        fn = getattr(obj, "reset_peak_memory", None)
        if fn is not None:
            try:
                fn()
                return True
            except Exception:
                return False
    return False


def _clear():
    try:
        mx.clear_cache()
    except Exception:
        try:
            mx.metal.clear_cache()
        except Exception:
            pass


def _peak_gb():
    p = _mx("get_peak_memory")
    return (p / (1 << 30)) if p else 0.0


def _rss_gb():
    if _PROC is not None:
        try:
            return _PROC.memory_info().rss / (1 << 30)
        except Exception:
            pass
    return 0.0


# --- cache build/copy (mirrors server _make_cache / _copy_cache) ----------- #
def make_cache(model, kv_bits, group):
    c = make_prompt_cache(model)
    if kv_bits:
        c = [e.to_quantized(group_size=group, bits=kv_bits)
             if isinstance(e, KVCache) else e for e in c]
    return c


def state_arrays(cache):
    out = []

    def rec(x):
        if isinstance(x, mx.array):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                rec(y)

    for c in cache:
        try:
            rec(c.state)
        except Exception:
            pass
    return out


def copy_cache(model, src, kv_bits, group):
    dst = make_cache(model, kv_bits, group)
    for s, d in zip(src, dst):
        try:
            d.state = s.state
        except Exception:
            pass
        try:
            d.meta_state = s.meta_state
        except Exception:
            pass
    arrs = state_arrays(dst)
    if arrs:
        mx.eval(arrs)
    return dst


def prefill(model, cache, toks, step):
    out = None
    i = 0
    while i < len(toks):
        out = model(mx.array(toks[i:i + step])[None], cache=cache)
        mx.eval(state_arrays(cache))
        _clear()
        i += step
    logits = out[:, -1, :]
    mx.eval(logits)
    return logits


def make_tokens(tok, n):
    ids = tok.encode(_BASE)
    if not ids:
        ids = [1, 2, 3, 4]
    reps = (n // len(ids)) + 1
    return (ids * reps)[:n]


def run_cell(model, tok, ctx_len, kv_bits, prefill_step, n_decode, group):
    toks = make_tokens(tok, ctx_len)

    _clear()
    _reset_peak()

    # Prefill (the snapshot build).
    t0 = time.perf_counter()
    last = prefill(model, model_cache := make_cache(model, kv_bits, group), toks,
                   prefill_step)
    t_prefill = time.perf_counter() - t0
    peak_after_prefill = _peak_gb()

    # Working copy — generation runs on this in the server, so the copy is part
    # of the real peak.
    tc = time.perf_counter()
    work = copy_cache(model, model_cache, kv_bits, group)
    t_copy = time.perf_counter() - tc
    peak_after_copy = _peak_gb()

    # Decode n_decode tokens greedily on the copy; TTFT is end-to-end.
    logits = last
    t_first = None
    td = time.perf_counter()
    for step_i in range(n_decode):
        tid = int(mx.argmax(logits[0]).item())
        o = model(mx.array([tid])[None], cache=work)
        logits = o[:, -1, :]
        mx.eval(logits)
        if step_i == 0:
            t_first = time.perf_counter()
    t_decode = time.perf_counter() - td
    peak_final = _peak_gb()

    ttft = (t_first - t0) if t_first else float("nan")
    prefill_tps = ctx_len / t_prefill if t_prefill > 0 else 0.0
    decode_tps = (n_decode - 1) / (t_decode - (t_first - td)) \
        if (t_first and t_decode > (t_first - td)) else 0.0

    peak = max(peak_after_prefill, peak_after_copy, peak_final)
    rss = _rss_gb()

    del model_cache, work
    _clear()

    return {
        "ctx": ctx_len, "kv_bits": kv_bits, "prefill_step": prefill_step,
        "prefill_tps": round(prefill_tps, 1),
        "ttft_s": round(ttft, 2),
        "decode_tps": round(decode_tps, 1),
        "peak_gb": round(peak, 2),
        "rss_gb": round(rss, 2),
        "copy_s": round(t_copy, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "QWEN_MLX", "mlx-community/Qwen3.6-27B-OptiQ-4bit"))
    ap.add_argument("--contexts", default="4096,16384,32768,65536",
                    help="comma-separated prompt token lengths")
    ap.add_argument("--kv-bits", default="8,4", help="comma-separated: 8,4 (0=fp16)")
    ap.add_argument("--prefill-steps", default="2048,4096", help="comma-separated")
    ap.add_argument("--kv-group", type=int,
                    default=int(os.environ.get("QWEN_KV_GROUP", "64")))
    ap.add_argument("--decode", type=int, default=64, help="tokens to decode/cell")
    ap.add_argument("--mem-limit", type=float, default=40.0,
                    help="flag cells whose peak exceeds this many GB")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    kv_bits_list = [int(x) for x in args.kv_bits.split(",") if x.strip()]
    steps = [int(x) for x in args.prefill_steps.split(",") if x.strip()]

    print(f"[ab] loading {args.model} ...")
    model, tok = load(args.model)
    if not _reset_peak():
        print("[ab] NOTE: reset_peak_memory unavailable — peak is GLOBAL "
              "high-water, not per-cell (treat as an upper bound).")
    print(f"[ab] weights resident: peak~{_peak_gb():.1f}GB rss~{_rss_gb():.1f}GB")
    print(f"[ab] sweep: contexts={contexts} kv_bits={kv_bits_list} "
          f"prefill_steps={steps} decode={args.decode} limit={args.mem_limit}GB\n")

    rows = []
    configs = list(itertools.product(kv_bits_list, steps))
    total = len(configs) * len(contexts)
    done = 0
    for kv_bits, step in configs:
        label = f"kv={kv_bits or 'fp16'}  prefill_step={step}"
        print(f"=== {label} " + "=" * (40 - len(label)))
        print(f"  {'ctx':>7} | {'prefill t/s':>11} | {'TTFT s':>7} | "
              f"{'decode t/s':>10} | {'peak GB':>8} | {'rss GB':>7} | fit?")
        print("  " + "-" * 74)
        for ctx in contexts:
            done += 1
            sys.stdout.write(f"  [{done}/{total}] running ctx={ctx} ...\r")
            sys.stdout.flush()
            try:
                r = run_cell(model, tok, ctx, kv_bits, step, args.decode,
                             args.kv_group)
            except Exception as e:
                print(f"  ctx={ctx}: FAILED ({type(e).__name__}: {e})")
                rows.append({"ctx": ctx, "kv_bits": kv_bits,
                             "prefill_step": step, "error": str(e)})
                _clear()
                continue
            r["label"] = label
            rows.append(r)
            fit = "OK" if r["peak_gb"] <= args.mem_limit else "OVER"
            print(f"  {r['ctx']:>7} | {r['prefill_tps']:>11} | {r['ttft_s']:>7} | "
                  f"{r['decode_tps']:>10} | {r['peak_gb']:>8} | {r['rss_gb']:>7} | "
                  f"{fit}")
        print()

    os.makedirs("ab_results", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    jpath = os.path.join("ab_results", f"{ts}.json")
    cpath = os.path.join("ab_results", f"{ts}.csv")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "mem_limit_gb": args.mem_limit,
                   "decode": args.decode, "results": rows}, f, indent=2)
    cols = ["label", "ctx", "kv_bits", "prefill_step", "prefill_tps", "ttft_s",
            "decode_tps", "peak_gb", "rss_gb", "copy_s", "error"]
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[ab] wrote {jpath} and {cpath}")

    over = [r for r in rows if r.get("peak_gb", 0) > args.mem_limit]
    if over:
        print(f"\n[ab] {len(over)} cell(s) EXCEED {args.mem_limit}GB:")
        for r in over:
            print(f"     kv={r['kv_bits']} step={r['prefill_step']} "
                  f"ctx={r['ctx']} -> peak {r['peak_gb']}GB")
        print("[ab] those configs would be refused/evicted under "
              "QWEN_MEM_BUDGET_GB=40 in production.")
    else:
        print(f"\n[ab] all cells fit under {args.mem_limit}GB.")


if __name__ == "__main__":
    sys.exit(main())
