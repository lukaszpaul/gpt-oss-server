#!/usr/bin/env python3
"""
test_cache_roundtrip.py — GATE for the disk-backed snapshot cache.

Proves that save_prompt_cache -> load_prompt_cache reproduces THIS model's
quantized-KV + recurrent (hybrid-attention) cache state EXACTLY, so reloading a
spilled conversation snapshot from disk yields identical generation to keeping
it resident. If this fails, QWEN_DISK_CACHE must stay OFF: a bad round-trip
corrupts context silently (plausible-looking garbage), which is worse than a
cache miss.

WHAT IT CHECKS
  1. STATE EQUALITY  — every cache layer's arrays after load match the live
     snapshot (exact for integer/quantized arrays, allclose for floats). This
     localizes which layer TYPE (quantized self_attn vs recurrent linear_attn)
     diverged, before generation muddies it.
  2. END-TO-END      — the real server path: load snapshot -> prefill a small
     suffix -> greedily generate N tokens, compared token-for-token (and on the
     first-step logits) against doing the same on the resident snapshot. This is
     the path that an offset/recurrent-state bug only reveals once you append
     tokens past the restored boundary.

RUN (on the Apple-Silicon box, same env as the server)
  export QWEN_MLX="mlx-community/Qwen3.6-27B-OptiQ-4bit"   # match the server
  export QWEN_KV_BITS=8                                    # match the server
  python test_cache_roundtrip.py

Exit code 0 = round-trip sound, safe to set QWEN_DISK_CACHE=1. Non-zero = do not
enable.
"""

import os
import sys
import tempfile

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import (
    KVCache,
    make_prompt_cache,
    save_prompt_cache,
    load_prompt_cache,
)

MODEL_PATH = os.environ.get("QWEN_MLX", "mlx-community/Qwen3.6-27B-OptiQ-4bit")
KV_BITS = int(os.environ.get("QWEN_KV_BITS", "8"))
KV_GROUP_SIZE = int(os.environ.get("QWEN_KV_GROUP", "64"))
PREFILL_STEP = int(os.environ.get("QWEN_PREFILL_STEP", "2048"))
N_GEN = int(os.environ.get("ROUNDTRIP_N_GEN", "40"))   # tokens to compare
N_SUFFIX = int(os.environ.get("ROUNDTRIP_SUFFIX", "8"))  # held-out tail tokens

# A prompt long enough to exercise multi-chunk prefill and a non-trivial KV.
PROMPT = (
    "You are a meticulous software engineer. Carefully reason step by step.\n\n"
    "Here is a Python function that needs review:\n\n"
    "def fib(n):\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n\n"
    "Explain precisely what this returns for n=10, then describe the time and "
    "space complexity, then suggest one improvement. Be thorough and concrete, "
    "and walk through the loop iteration by iteration before giving the answer. "
) * 6


def make_cache(model):
    """Mirror the server's _make_cache: quantize the growing self_attn KV layers,
    leave the recurrent linear_attn layers untouched."""
    c = make_prompt_cache(model)
    if KV_BITS:
        c = [e.to_quantized(group_size=KV_GROUP_SIZE, bits=KV_BITS)
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


def feed(model, cache, toks):
    """Prefill toks into cache in chunks; return last-position logits."""
    out = None
    i = 0
    while i < len(toks):
        out = model(mx.array(toks[i:i + PREFILL_STEP])[None], cache=cache)
        mx.eval(state_arrays(cache))
        i += PREFILL_STEP
    logits = out[:, -1, :]
    mx.eval(logits)
    return logits


def greedy(model, cache, last_logits, n):
    """Greedy (argmax) decode n tokens, mutating `cache`. Returns the token ids
    and the first-step logits (for a numerical comparison argmax can mask)."""
    ids = []
    logits = last_logits
    first = mx.array(last_logits)
    for step in range(n):
        tid = int(mx.argmax(logits[0]).item())
        ids.append(tid)
        out = model(mx.array([tid])[None], cache=cache)
        logits = out[:, -1, :]
        mx.eval(logits)
        if step == 0:
            first = mx.array(last_logits)
    return ids, first


def compare_states(ref, loaded):
    """Per-layer array equality between the LIVE snapshot and the reloaded one.
    Must be called BEFORE any generation: recurrent (ArraysCache) layers update
    their state in place, so comparing after a decode would show false diffs.

    Returns (value_bad, shape_bad). Only VALUE differences are real corruption;
    SHAPE differences are usually benign quantized-KV buffer padding (the live
    cache preallocates to a step boundary, the reload may trim) and are reported
    but not counted as failures — the end-to-end generation check is the
    ground truth for those."""
    value_bad, shape_bad = 0, 0
    for i, (a, b) in enumerate(zip(ref, loaded)):
        aa, bb = state_arrays([a]), state_arrays([b])
        tname = type(a).__name__
        if len(aa) != len(bb):
            print(f"  layer {i} ({tname}): array count {len(aa)} != {len(bb)}")
            value_bad += 1
            continue
        for j, (x, y) in enumerate(zip(aa, bb)):
            if x.shape != y.shape or x.dtype != y.dtype:
                print(f"  layer {i} ({tname}) arr {j}: SHAPE/DTYPE "
                      f"{x.shape}/{x.dtype} != {y.shape}/{y.dtype} (benign?)")
                shape_bad += 1
                continue
            if x.dtype in (mx.float16, mx.bfloat16, mx.float32):
                ok = bool(mx.allclose(x.astype(mx.float32), y.astype(mx.float32),
                                      atol=1e-3, rtol=1e-3).item())
            else:                                   # uint32-packed quant, ints
                ok = bool(mx.all(x == y).item())
            if not ok:
                print(f"  layer {i} ({tname}) arr {j}: VALUES DIFFER "
                      f"(dtype={x.dtype}, shape={x.shape})")
                value_bad += 1
    return value_bad, shape_bad


def main():
    print(f"[test] loading {MODEL_PATH} (kv={'%d-bit' % KV_BITS if KV_BITS else 'fp16'}) ...")
    model, tok = load(MODEL_PATH)
    toks = tok.encode(PROMPT)
    if len(toks) <= N_SUFFIX + 4:
        print("[test] prompt too short; increase PROMPT")
        return 2
    snap_toks, suffix = toks[:-N_SUFFIX], toks[-N_SUFFIX:]
    print(f"[test] prompt={len(toks)}t  snapshot={len(snap_toks)}t  "
          f"suffix={len(suffix)}t  gen={N_GEN}t")

    # Build + prefill the snapshot, spill to disk, reload.
    cache = make_cache(model)
    feed(model, cache, snap_toks)

    tmp = os.path.join(tempfile.gettempdir(), "qwen_roundtrip.safetensors")
    save_prompt_cache(tmp, cache, metadata={"n": str(len(snap_toks))})
    sz = os.path.getsize(tmp)
    print(f"[test] saved snapshot -> {tmp} ({sz / (1 << 20):.1f} MB)")

    loaded = load_prompt_cache(tmp)
    mx.eval(state_arrays(loaded))

    # (1) Compare the live snapshot vs the reloaded one BEFORE any generation —
    # recurrent layers mutate state in place, so this must precede the decode.
    print("[test] (1) comparing reloaded snapshot vs live snapshot ...")
    value_bad, shape_bad = compare_states(cache, loaded)
    if value_bad == 0:
        print(f"       OK — no value differences"
              + (f" ({shape_bad} benign shape/padding diff(s))" if shape_bad
                 else "") + ".")
    else:
        print(f"       {value_bad} array(s) differ in VALUES "
              f"(+{shape_bad} shape diff(s)).")

    # (2) Same suffix prefill + greedy decode on each; compared token-for-token.
    ram_logits = feed(model, cache, suffix)
    ram_ids, ram_first = greedy(model, cache, ram_logits, N_GEN)
    disk_logits = feed(model, loaded, suffix)
    disk_ids, disk_first = greedy(model, loaded, disk_logits, N_GEN)

    print("[test] (2) comparing end-to-end generation (load + suffix + decode) ...")
    ids_match = ram_ids == disk_ids
    logit_close = bool(mx.allclose(ram_first.astype(mx.float32),
                                   disk_first.astype(mx.float32),
                                   atol=1e-2, rtol=1e-2).item())
    if ids_match:
        print(f"       OK — {N_GEN}/{N_GEN} generated tokens identical.")
    else:
        first_div = next((i for i, (a, b) in enumerate(zip(ram_ids, disk_ids))
                          if a != b), 0)
        print(f"       FAIL — diverges at generated token {first_div}.")
        print(f"         ram : {ram_ids[:12]}")
        print(f"         disk: {disk_ids[:12]}")
    print(f"       first-step logits allclose: {logit_close}")

    try:
        os.remove(tmp)
    except Exception:
        pass

    # Generation equivalence is the ground truth; value-level state diffs back it
    # up. Shape/padding diffs alone (shape_bad) do not fail the round-trip.
    passed = ids_match and logit_close and value_bad == 0
    print()
    if passed:
        extra = f" ({shape_bad} benign shape diff(s) ignored)" if shape_bad else ""
        print(f"[test] PASS — round-trip is sound{extra}. "
              f"Safe to set QWEN_DISK_CACHE=1 / QWEN_FRONT_SNAPSHOT=1.")
        return 0
    print("[test] FAIL — generation diverged or cache values differ; do NOT "
          "enable the disk store on this build.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
