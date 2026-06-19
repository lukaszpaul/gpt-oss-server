#!/usr/bin/env python3
"""
gpt-server-27b-repomap.py — PROTOTYPE of gpt-server-27b.py with REPO MAP
injection (Tier-1 context priming). Everything from the base server is intact;
this adds an offline structural map of the *workspace the request comes from*
into the snapshot-cached stable prefix, so the model knows where code lives
without burning slow-prefill tool-call round-trips exploring it.

REPO MAP (what's new vs gpt-server-27b.py)
  PROBLEM. On a large codebase the model wastes ~100k tokens grepping/reading
  just to orient itself, and every one of those tool calls is a slow dense
  prefill. We replace exploration with a precomputed map handed to it up front.

  WHERE FROM (the "where are we prompting from" problem). The server runs in one
  place (e.g. gpt-oss-server) but Copilot queries come from a DIFFERENT VS Code
  window (e.g. Desktop/dev-tools). VS Code embeds the open workspace root in the
  first user message's <workspace_info> block:
      I am working in a workspace with the following folders:
      - c:\\Users\\lukep\\Desktop\\dev-tools
  We parse THAT path (never the server's cwd), build/lookup the map for it, and
  inject it. Different workspace -> different path -> different rendered map ->
  different token prefix -> its own snapshot slot, automatically.

  WHERE INJECTED (so it caches). <workspace_info> sits BEFORE the <context>
  marker, so the injected <repoMap> block falls inside the snapshot-cached
  stable prefix: prefilled ONCE per repo instance and reused every turn, exactly
  like the system prompt. It only re-prefills when the map text actually changes
  (new commit / new symbols).

  TWO CACHES, both required:
    1. Repo-map cache (this file): the rendered map is cached in memory + on
       disk (./repomaps), keyed by the absolute workspace path, validated by git
       HEAD (falls back to a TTL when the workspace isn't a git repo). So we walk
       the filesystem rarely, not per request, and the rendered string stays
       BYTE-IDENTICAL across turns — which is what keeps cache #2 valid.
    2. The server's existing KV snapshot cache, which reuses that identical
       prefix.

  Knobs: QWEN_REPOMAP (on/off), QWEN_REPOMAP_DIR, QWEN_REPOMAP_TTL,
  QWEN_REPOMAP_MAX_CHARS, QWEN_REPOMAP_MAX_FILES, QWEN_REPOMAP_SYMBOLS,
  QWEN_REPOMAP_IGNORE. See the config block below.

----------------------------------------------------------------------------
BASE SERVER (unchanged) — OpenAI-compatible server for Qwen3.6-27B (MLX OptiQ
4-bit) on Apple Silicon, drop-in replacement for qwen-server-gguf.py as a GitHub
Copilot Custom Endpoint (same URL, same tool calling, same <think> handling).

WHY MLX (vs the llama-cpp-python version)
  The GGUF server's prefix reuse was implicit (one hidden slot inside
  llama-cpp-python) and was observed re-prefilling the full ~30k Copilot
  prompt every turn despite >95% token-level prefix match. Here the prompt
  cache is EXPLICIT and owned by this script:

  1. SLOT CACHE. QWEN_CACHE_SLOTS (default 3) independent prompt caches.
     Each request is routed to the slot whose snapshot is the longest exact
     prefix of the new prompt, so a Copilot side request (title gen,
     summarization) lands in its own slot instead of evicting the agent
     loop's 30k-token prefix.

  2. SNAPSHOT + DELTA PREFILL. Qwen3.6-27B is dense (no MoE experts) but
     still a HYBRID-ATTENTION model (qwen3_5 arch, 64 layers alternating
     linear_attn / self_attn): most layers are linear attention whose
     cache is a fixed-size recurrent state
     (mlx_lm ArraysCache, no .offset) that can NEVER be trimmed back to an
     earlier token position. So trim-based reuse is impossible; instead each
     slot stores a SNAPSHOT of the cache taken right after prefilling the
     prompt (minus a small holdback tail), BEFORE generation touches it.
     Next turn: copy snapshot -> working cache (GPU state copy, tens of
     ms), prefill only the new suffix, generate on the working copy, and
     advance the snapshot.

  3. VISIBILITY. Every request logs: slot id, prompt tokens, reused tokens,
     suffix actually prefilled, TTFT, prefill t/s (from MLX itself), and
     decode t/s. If reuse ever drops, you'll see exactly when and how much.

  KV cache is quantized to 8-bit (QWEN_KV_BITS / QWEN_KV_GROUP). Hybrid
  attention already keeps KV small — most layers hold a fixed-size
  recurrent state, only the self_attn layers grow with tokens — and 8-bit
  halves the growing part vs fp16, which matters at 128k context on 64GB
  next to ~17.5GB of OptiQ 4-bit weights. Sampling matches the GGUF
  server, including an additive OpenAI-style presence penalty implemented as
  a custom logits processor (mlx_lm only ships repetition_penalty natively;
  presence is Qwen3.6's recommended anti-loop knob, so it's reimplemented).

THREADING
  Same pattern as the gpt-oss MLX server: ALL Metal work pinned to one
  worker thread (ThreadPoolExecutor(max_workers=1)); SSE generators only
  drain a queue. MLX is not safe for concurrent graph eval from multiple
  threads — this is the fix for the Metal threading crash seen before.

RUN
  python3 -m venv ~/mlx-env && source ~/mlx-env/bin/activate
  pip install -U mlx-lm fastapi "uvicorn[standard]"
  export QWEN_MLX="mlx-community/Qwen3.6-27B-OptiQ-4bit"       # or a local dir
  python gpt-server-27b.py                                     # http://127.0.0.1:8000

  Still worth keeping at the OS level (weight-eviction insurance):
    sudo sysctl iogpu.wired_limit_mb=57344
  Optionally also QWEN_WIRED_GB=56 to ask MLX to wire its buffers.

VS CODE (chatLanguageModels.json): unchanged — point `url` at
  http://localhost:8000/v1/chat/completions , "toolCalling": true, "thinking": true
"""

import os
import re
import json
import time
import uuid
import asyncio
import queue
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # optional; gives current process RSS
    _PROC = psutil.Process()
except Exception:
    psutil = None
    _PROC = None

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import (
    KVCache,
    make_prompt_cache,
    can_trim_prompt_cache,
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get("QWEN_MLX", "mlx-community/Qwen3.6-27B-OptiQ-4bit")
MODEL_ID = os.environ.get("QWEN_ID", "qwen3.6-27b")
PORT = int(os.environ.get("QWEN_PORT", "8000"))
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "4096"))

# Prefill chunk size MLX pushes through Metal per step. 2048 mirrors the GGUF
# server's n_ubatch; try 4096 on M4 Max (cheap A/B, watch the prefill t/s log).
PREFILL_STEP = int(os.environ.get("QWEN_PREFILL_STEP", "2048"))

# Prompt-cache slots. 3 = agent loop + two side-request streams (Copilot can
# fire title gen AND summaries between agent turns) without mutual eviction.
# Each slot holds quantized KV for its full prompt.
CACHE_SLOTS = int(os.environ.get("QWEN_CACHE_SLOTS", "3"))
# Snapshots exclude the last N prompt tokens. The prompt tail (generation
# prompt, "<think>" opener) is re-rendered differently next turn, and a
# recurrent state can't roll back, so we simply never cache the volatile
# tail. Used as the FALLBACK when QWEN_SNAPSHOT_MARKER isn't found.
SNAPSHOT_HOLDBACK = int(os.environ.get("QWEN_SNAPSHOT_HOLDBACK", "64"))

# Structural snapshot boundary. The snapshot is taken right before the LAST
# occurrence of this marker, so the stable prefix (system prompt + workspace
# info) is cached while the volatile per-turn tail after it (date, terminals,
# editor state, the user request, generation prompt) is recomputed fresh every
# turn. Default "<context>" is VS Code Copilot's per-turn context block. Unlike
# a fixed token holdback, this auto-fits however long the volatile tail grows —
# a big terminal dump can't silently push the snapshot into changing content.
# Empty string disables it and falls back to QWEN_SNAPSHOT_HOLDBACK.
SNAPSHOT_MARKER = os.environ.get("QWEN_SNAPSHOT_MARKER", "<context>")

# KV cache quantization for the growing self_attn layers (the recurrent
# linear-attn layers are fixed-size state and unaffected). 8-bit halves KV
# memory vs fp16 with negligible quality loss; set QWEN_KV_BITS=4 to halve
# again if 128k contexts still squeeze, or 0 for fp16.
KV_BITS = int(os.environ.get("QWEN_KV_BITS", "8"))
KV_GROUP_SIZE = int(os.environ.get("QWEN_KV_GROUP", "64"))

# Cache diagnostics: when a request diverges from its best-matching slot,
# print WHERE it diverged and the differing text on both sides. This is the
# tool for hunting prompt churn (timestamps, shuffled tools, editor state).
CACHE_DEBUG = os.environ.get("QWEN_CACHE_DEBUG", "1") == "1"

# Memory diagnostics: log MLX allocator stats (active / cache pool / peak) and
# process RSS at each phase of a request (start, after prefill+copy, after
# generation). At 128k context on 64GB this is the tool for catching the
# unified-memory exhaustion that crashes Python: each slot holds a full
# quantized-KV snapshot AND generation runs on a live COPY of it, so peak
# transiently doubles the largest slot. Watch "peak" approach the wired/RAM
# ceiling right before a crash. Set QWEN_MEM_LOG=0 to silence.
MEM_LOG = os.environ.get("QWEN_MEM_LOG", "1") == "1"

# Copilot is not guaranteed to send the tools array in a stable order, and
# tools render near the TOP of the Qwen prompt — a shuffled tool list breaks
# ALL prefix reuse by itself. Sorting by function name makes the rendered
# prompt deterministic; the model just sees a consistent ordering.
SORT_TOOLS = os.environ.get("QWEN_SORT_TOOLS", "1") == "1"

# Prompt pinning: regex substitutions applied to the rendered prompt before
# tokenization, to neutralize volatile substrings (dates, timestamps, session
# ids) that Copilot embeds in the system prompt and that otherwise break
# prefix reuse across requests. The model SEES the pinned
# value, so only pin text whose exact value doesn't matter for coding.
# Format: JSON list of [pattern, replacement] pairs, e.g.
#   QWEN_PIN='[["Current date: [^\\n]+", "Current date: 2026-01-01"]]'
# Use the CACHE_DEBUG divergence log (cached:/new: lines) to find what to pin.
PIN_PATTERNS: List[Tuple["re.Pattern", str]] = []
_pin_raw = os.environ.get("QWEN_PIN", "")
if _pin_raw:
    try:
        for _pat, _repl in json.loads(_pin_raw):
            PIN_PATTERNS.append((re.compile(_pat), _repl))
        print(f"[qwen-server] prompt pinning: {len(PIN_PATTERNS)} pattern(s)")
    except Exception as _e:
        print(f"[qwen-server] QWEN_PIN parse failed, ignoring: {_e!r}")

# Optional: ask MLX to wire its GPU buffers (resists page eviction between
# turns). Set to e.g. 56 (GB). Requires the sysctl above to be raised first.
WIRED_GB = float(os.environ.get("QWEN_WIRED_GB", "0"))

# --- Memory governor ------------------------------------------------------- #
# Soft RAM ceiling for THIS process (GB). The governor keeps resident usage
# under it by evicting least-recently-used slot snapshots BEFORE prefilling a
# new request, and (with QWEN_MEM_STRICT on) refusing a prompt that still can't
# fit once every other slot is gone — a clean 503 instead of an OOM crash. On a
# 64GB box with ~20GB of background load, 40 leaves room for the OS plus the
# transient snapshot+copy double that generation needs. 0 = governor off.
MEM_BUDGET_GB = float(os.environ.get("QWEN_MEM_BUDGET_GB", "0"))
MEM_BUDGET_BYTES = int(MEM_BUDGET_GB * (1 << 30))

# When a request can't fit under the budget even after evicting every other
# slot: strict (1) refuses it with a 503; lenient (0) logs a warning and tries
# anyway. Default strict, since the whole point is a hard cap.
MEM_STRICT = os.environ.get("QWEN_MEM_STRICT", "1") == "1"

# Cap MLX's reusable buffer pool (GB). MLX retains freed buffers in this pool
# to avoid re-allocating; left uncapped it behaves like a slow leak — RSS
# climbs request over request ("worse after a few questions") even though the
# live arrays didn't grow. Capping it forces freed KV back to the OS. 0 =
# MLX default. A few GB is plenty of scratch; try 4–8.
CACHE_LIMIT_GB = float(os.environ.get("QWEN_CACHE_LIMIT_GB", "0"))

# Hard upper bound on prompt length (tokens). A prompt above this is refused
# with a clean error rather than allowed to OOM the box. 0 = no cap.
MAX_CONTEXT = int(os.environ.get("QWEN_MAX_CONTEXT", "0"))

# Seed for the self-calibrating KV-bytes-per-token estimate the governor uses
# to project a request's footprint. It is refined by direct allocator
# measurement after every prefill, so this only matters for the first request
# or two; the default is a conservative 8-bit-KV hybrid-attention figure.
KV_BYTES_PER_TOK = float(os.environ.get("QWEN_KV_BYTES_PER_TOK", "70000"))

# Global thinking kill-switch (same semantics as GGUF server).
THINKING_DEFAULT = os.environ.get("QWEN_THINKING", "1") == "1"

# Per-request chat logging into ./chats (same format as GGUF server).
CHAT_LOG = os.environ.get("QWEN_CHAT_LOG", "1") == "1"
CHAT_LOG_DIR = os.environ.get("QWEN_CHAT_LOG_DIR", os.path.join(os.getcwd(), "chats"))

# Qwen3.6 sampling defaults (coding-leaning, same as GGUF server).
DEF_TEMP = float(os.environ.get("QWEN_TEMP", "0.7"))
DEF_TOP_P = float(os.environ.get("QWEN_TOP_P", "0.8"))
DEF_TOP_K = int(os.environ.get("QWEN_TOP_K", "20"))
DEF_MIN_P = float(os.environ.get("QWEN_MIN_P", "0.0"))
DEF_REPEAT = float(os.environ.get("QWEN_REPEAT_PENALTY", "1.05"))
DEF_PRESENCE = float(os.environ.get("QWEN_PRESENCE_PENALTY", "1.5"))

# --- Repo map ------------------------------------------------------------- #
# Master switch. When on, every request whose <workspace_info> names a workspace
# root that exists on THIS host gets a <repoMap> block injected before <context>
# (i.e. into the snapshot-cached stable prefix).
REPOMAP_ENABLED = os.environ.get("QWEN_REPOMAP", "1") == "1"

# Where rendered maps are cached on disk (survives server restarts). One JSON per
# workspace root, named by a hash of the absolute path.
REPOMAP_DIR = os.environ.get("QWEN_REPOMAP_DIR", os.path.join(os.getcwd(), "repomaps"))

# How long a built map is trusted before we re-check the repo signature (git
# HEAD) / rebuild. Within this window we return the cached string WITHOUT even
# walking the tree, so the injected prefix is byte-stable and the KV snapshot
# stays valid no matter how many edits you make mid-session. Seconds.
REPOMAP_TTL = float(os.environ.get("QWEN_REPOMAP_TTL", "300"))

# Hard ceiling on the rendered map size (chars). The map shares the context
# budget, so cap it: lowest-ranked files are dropped (with an "N more omitted"
# note) until it fits. ~24k chars ≈ ~6k tokens.
REPOMAP_MAX_CHARS = int(os.environ.get("QWEN_REPOMAP_MAX_CHARS", "24000"))

# Stop walking after this many indexable files (protects against a pathological
# tree). Files past the cap simply aren't in the map.
REPOMAP_MAX_FILES = int(os.environ.get("QWEN_REPOMAP_MAX_FILES", "4000"))

# Per-file byte cap for symbol extraction (we only scan the head of each file).
REPOMAP_MAX_FILE_BYTES = int(os.environ.get("QWEN_REPOMAP_MAX_FILE_BYTES", "200000"))

# Extract top-level symbols (defs/classes/exports/types) per file, not just the
# tree. Symbols are what make the map actually useful for "where does X live".
REPOMAP_SYMBOLS = os.environ.get("QWEN_REPOMAP_SYMBOLS", "1") == "1"

# Max symbols listed per file (keeps fat files from dominating the budget).
REPOMAP_SYMS_PER_FILE = int(os.environ.get("QWEN_REPOMAP_SYMS_PER_FILE", "12"))

# Comma-separated extra directory names to ignore, on top of the built-in set.
REPOMAP_EXTRA_IGNORES = [d.strip() for d in
                         os.environ.get("QWEN_REPOMAP_IGNORE", "").split(",")
                         if d.strip()]

# All MLX/Metal work happens on this one thread. SSE generators only drain a
# queue (pure Python). Never call model code from another thread.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def _set_wired_limit():
    if not WIRED_GB:
        return
    nbytes = int(WIRED_GB * (1 << 30))
    for fn in ("set_wired_limit",):
        try:
            getattr(mx, fn)(nbytes)
            print(f"[qwen-server] MLX wired limit set to {WIRED_GB:g} GB")
            return
        except AttributeError:
            pass
        except Exception as e:
            print(f"[qwen-server] could not set wired limit: {e!r}")
            return
    try:
        mx.metal.set_wired_limit(nbytes)  # older MLX
        print(f"[qwen-server] MLX wired limit set to {WIRED_GB:g} GB (metal API)")
    except Exception as e:
        print(f"[qwen-server] wired limit unavailable: {e!r}")


def _set_memory_limits():
    """Cap MLX's buffer pool (and hint its allocation ceiling) so freed KV
    returns to the OS instead of inflating this process between requests.
    Version-tolerant: these setters moved from mx.metal.* to top-level mx.*
    across MLX releases, so probe both and never let a missing API be fatal."""
    def _call(name, nbytes) -> bool:
        for obj in (mx, getattr(mx, "metal", None)):
            if obj is None:
                continue
            fn = getattr(obj, name, None)
            if fn is None:
                continue
            try:
                fn(nbytes)
                return True
            except Exception as e:
                print(f"[qwen-server] {name}({nbytes}) failed: {e!r}")
                return False
        return False

    if CACHE_LIMIT_GB:
        if _call("set_cache_limit", int(CACHE_LIMIT_GB * (1 << 30))):
            print(f"[qwen-server] MLX cache pool capped at {CACHE_LIMIT_GB:g} GB")
        else:
            print("[qwen-server] set_cache_limit unavailable in this MLX build")
    if MEM_BUDGET_BYTES:
        # The budget hint's exact signature varies by version (some take a
        # relaxed flag); try the known forms, tolerate absence. The real
        # enforcement is the eviction governor below — this is belt-and-braces.
        for args in ((MEM_BUDGET_BYTES,), (MEM_BUDGET_BYTES, True)):
            try:
                fn = getattr(mx, "set_memory_limit", None)
                if fn is None:
                    break
                fn(*args)
                print(f"[qwen-server] MLX memory limit hint = {MEM_BUDGET_GB:g} GB")
                break
            except TypeError:
                continue
            except Exception as e:
                print(f"[qwen-server] set_memory_limit failed: {e!r}")
                break


def _load():
    _set_wired_limit()
    _set_memory_limits()
    print(f"[qwen-server] loading MLX model {MODEL_PATH} ...")
    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer


model, tok = GPU.submit(_load).result()


def _make_cache():
    """Fresh prompt cache with the growing self_attn KV layers quantized to
    KV_BITS. The recurrent linear-attn layers (ArraysCache) are fixed-size
    state and are left untouched — only KVCache entries grow with tokens."""
    c = make_prompt_cache(model)
    if KV_BITS:
        c = [e.to_quantized(group_size=KV_GROUP_SIZE, bits=KV_BITS)
             if isinstance(e, KVCache) else e for e in c]
    return c


_probe = _make_cache()
print(f"[qwen-server] cache layer types: "
      f"{sorted(set(type(c).__name__ for c in _probe))}, "
      f"trimmable={can_trim_prompt_cache(_probe)} -> snapshot caching")
del _probe
print(f"[qwen-server] ready. model id: {MODEL_ID} "
      f"(slots={CACHE_SLOTS}, prefill_step={PREFILL_STEP}, "
      f"snapshot={('marker %r' % SNAPSHOT_MARKER) if SNAPSHOT_MARKER else 'holdback'}"
      f"+holdback={SNAPSHOT_HOLDBACK}, "
      f"kv={'%d-bit/g%d' % (KV_BITS, KV_GROUP_SIZE) if KV_BITS else 'fp16'})")
if MEM_BUDGET_BYTES or CACHE_LIMIT_GB or MAX_CONTEXT:
    print(f"[qwen-server] governor: budget={MEM_BUDGET_GB:g}GB "
          f"strict={MEM_STRICT} cache_pool_cap="
          f"{('%gGB' % CACHE_LIMIT_GB) if CACHE_LIMIT_GB else 'off'} "
          f"max_ctx={MAX_CONTEXT or 'off'} "
          f"seed~{int(KV_BYTES_PER_TOK)//1000}KB/tok")
else:
    print("[qwen-server] governor: OFF (set QWEN_MEM_BUDGET_GB to enable)")
if MEM_LOG and psutil is None:
    print("[qwen-server] mem: psutil not installed; rss is peak-only via "
          "resource.getrusage (pip install psutil for live rss)")
if REPOMAP_ENABLED:
    print(f"[qwen-server] repo map: ON (dir={REPOMAP_DIR}, ttl={REPOMAP_TTL:g}s, "
          f"max={REPOMAP_MAX_CHARS}c/{REPOMAP_MAX_FILES}f, "
          f"symbols={'on' if REPOMAP_SYMBOLS else 'off'}) — "
          f"injects per-workspace map from <workspace_info>")
else:
    print("[qwen-server] repo map: OFF (set QWEN_REPOMAP=1 to enable)")

app = FastAPI()


# --------------------------------------------------------------------------- #
# Prompt-cache slots — SNAPSHOT design (required for hybrid-attention models)
#
# Each slot stores a snapshot of the cache state for a prompt prefix, taken
# BEFORE any generation. Reuse = copy snapshot to a working cache, prefill
# only the suffix, generate on the copy. The snapshot itself is never
# contaminated by generated tokens, so it never needs trimming — which is
# the operation Qwen3.6's recurrent-state layers cannot do.
# --------------------------------------------------------------------------- #
class _Slot:
    __slots__ = ("idx", "cache", "tokens", "last_used")

    def __init__(self, idx: int):
        self.idx = idx
        self.cache = _make_cache()
        self.tokens: List[int] = []   # ledger of tokens the cache contains
        self.last_used = 0.0

    def reset(self):
        self.cache = _make_cache()
        self.tokens = []


_SLOTS: List[_Slot] = []


def _common_prefix(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _log_divergence(slot: "_Slot", tokens: List[int], k: int):
    """Print where the new prompt stops matching a snapshot it could not
    reuse, with the text on both sides of the split. One glance identifies
    the churn source: a timestamp, shuffled tool defs, editor state, etc."""
    if k >= len(slot.tokens):
        return  # snapshot is a clean prefix — nothing diverged
    try:
        lo = max(0, k - 15)
        cached = tok.decode(slot.tokens[lo:k + 40])
        new = tok.decode(tokens[lo:k + 40])
        print(f"[qwen-server] slot={slot.idx} diverges INSIDE snapshot at "
              f"token {k}/{len(slot.tokens)} (snapshot unusable):\n"
              f"  cached: {cached!r}\n"
              f"  new:    {new!r}")
    except Exception:
        pass


def _acquire_slot(tokens: List[int]) -> Tuple[_Slot, int]:
    """Pick the slot whose snapshot is the longest EXACT prefix of the new
    prompt. A partially-matching snapshot is unusable for a recurrent state
    (no rollback), so it counts as zero. Worker thread only."""
    best, best_usable = None, 0
    best_partial, best_partial_k = None, -1
    for s in _SLOTS:
        if not s.tokens:
            continue
        k = _common_prefix(s.tokens, tokens)
        usable = len(s.tokens) if (k == len(s.tokens) and k < len(tokens)) else 0
        if usable > best_usable:
            best, best_usable = s, usable
        if k > best_partial_k:
            best_partial, best_partial_k = s, k
    if best is not None:
        return best, best_usable
    if CACHE_DEBUG and best_partial is not None:
        _log_divergence(best_partial, tokens, best_partial_k)
    if len(_SLOTS) < CACHE_SLOTS:
        s = _Slot(len(_SLOTS))
        _SLOTS.append(s)
        return s, 0
    s = min(_SLOTS, key=lambda x: x.last_used)   # LRU recycle, cold start
    s.reset()
    return s, 0


def _state_arrays(cache_list) -> List["mx.array"]:
    """Collect every mx.array leaf in a prompt cache's state (handles
    KVCache pairs, ArraysCache lists, and None entries alike)."""
    out: List[mx.array] = []

    def rec(x):
        if isinstance(x, mx.array):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                rec(y)

    for c in cache_list:
        try:
            rec(c.state)
        except Exception:
            pass
    return out


def _copy_cache(src):
    """Independent copy of a prompt cache via the .state/.meta_state API —
    the same mechanism mlx_lm's cache (de)serialization uses, so it works for
    (Quantized)KVCache and recurrent ArraysCache entries alike. Evaluated
    immediately so the copy owns materialized data before the source mutates
    further."""
    dst = _make_cache()
    for s, d in zip(src, dst):
        try:
            d.state = s.state
        except Exception:
            pass  # empty cache entries have no state to copy yet
        try:
            d.meta_state = s.meta_state
        except Exception:
            pass
    arrs = _state_arrays(dst)
    if arrs:
        mx.eval(arrs)
    return dst


def _clear_metal_cache():
    try:
        mx.clear_cache()
    except Exception:
        try:
            mx.metal.clear_cache()
        except Exception:
            pass


def _mx_mem(name: str) -> Optional[int]:
    """Read one MLX allocator counter (bytes) across MLX versions: the getters
    moved from mx.metal.* to top-level mx.* in newer releases."""
    for obj in (mx, getattr(mx, "metal", None)):
        if obj is None:
            continue
        fn = getattr(obj, name, None)
        if fn is not None:
            try:
                return int(fn())
            except Exception:
                return None
    return None


def _rss_bytes() -> Optional[int]:
    """Current resident set size of this process, in bytes (unified memory on
    Apple Silicon, so this includes GPU buffers MLX has wired/allocated)."""
    if _PROC is not None:
        try:
            return int(_PROC.memory_info().rss)
        except Exception:
            pass
    try:
        import resource  # macOS: ru_maxrss is bytes (peak, not current)
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


def _gb(n: Optional[int]) -> str:
    return f"{n / (1 << 30):.2f}GB" if n is not None else "n/a"


def _log_mem(phase: str, slot_idx: Optional[int] = None):
    """One memory snapshot line. `active` = live MLX arrays, `cache` = MLX's
    reusable buffer pool, `peak` = high-water mark since reset, `rss` = whole
    process. The gap (rss - active - cache) is non-MLX/python overhead."""
    if not MEM_LOG:
        return
    active = _mx_mem("get_active_memory")
    cache = _mx_mem("get_cache_memory")
    peak = _mx_mem("get_peak_memory")
    rss = _rss_bytes()
    where = f"slot={slot_idx} " if slot_idx is not None else ""
    print(f"[qwen-server] mem {where}[{phase}] "
          f"active={_gb(active)} cache={_gb(cache)} "
          f"peak={_gb(peak)} rss={'~' if _PROC is None else ''}{_gb(rss)} "
          f"slots={len(_SLOTS)}")


# --------------------------------------------------------------------------- #
# Memory governor — keep resident usage under MEM_BUDGET_BYTES
#
# Python can't impose a true OS ceiling on unified memory, so instead we (a)
# cap MLX's buffer pool at startup (above) so freed KV returns to the OS, and
# (b) before prefilling each request, PROJECT this request's footprint and
# evict least-recently-used slot snapshots until it fits — refusing outright
# if it can't fit even with every other slot gone. The per-token KV cost is
# measured live and refined after every prefill, so the projection tracks the
# real model/quantization instead of a guess.
# --------------------------------------------------------------------------- #
class MemoryBudgetExceeded(Exception):
    """Raised when a request cannot be served under MEM_BUDGET_BYTES."""


_BYTES_PER_TOK = KV_BYTES_PER_TOK   # EMA, refined by _update_bytes_per_tok


def _used_bytes() -> int:
    """Best estimate of this process's current resident footprint. Live RSS
    when psutil is present (truest — unified memory includes GPU buffers),
    else MLX's own active + pool counters."""
    if _PROC is not None:
        rss = _rss_bytes()
        if rss is not None:
            return rss
    a = _mx_mem("get_active_memory") or 0
    c = _mx_mem("get_cache_memory") or 0
    return a + c


def _est_kv_bytes(n_tokens: int) -> int:
    return int(max(0, n_tokens) * _BYTES_PER_TOK)


def _update_bytes_per_tok(active_before: Optional[int],
                          active_after: Optional[int], n_tokens: int):
    """Refine the per-token KV estimate from a real prefill's allocator delta.
    EMA so one noisy sample (e.g. a chunk freed mid-prefill) can't whipsaw the
    governor. Worker thread only."""
    global _BYTES_PER_TOK
    if active_before is None or active_after is None or n_tokens <= 0:
        return
    delta = active_after - active_before
    if delta <= 0:
        return
    _BYTES_PER_TOK = 0.7 * _BYTES_PER_TOK + 0.3 * (delta / n_tokens)


def _enforce_budget(target: "_Slot", n_prefill: int, snap_to: int):
    """Evict LRU non-target slots until this request is projected to fit under
    the budget. Raise MemoryBudgetExceeded if it still won't (strict), or warn
    and proceed (QWEN_MEM_STRICT=0). No-op when no budget is set. Worker thread
    only — it mutates _SLOTS and frees caches."""
    if not MEM_BUDGET_BYTES:
        return
    # New memory this request adds: KV for the suffix we prefill into the slot,
    # PLUS the live working COPY of the whole snapshot (snap_to tokens) that
    # generation runs on — that copy is the transient peak that tips the box.
    added = _est_kv_bytes(n_prefill) + _est_kv_bytes(snap_to)
    projected = _used_bytes() + added
    evicted = 0
    if projected > MEM_BUDGET_BYTES:
        victims = sorted((s for s in _SLOTS if s is not target and s.tokens),
                         key=lambda x: x.last_used)
        for s in victims:
            if projected <= MEM_BUDGET_BYTES:
                break
            freed = _est_kv_bytes(len(s.tokens))
            print(f"[qwen-server] governor: evicting slot={s.idx} "
                  f"({len(s.tokens)}t ~{_gb(freed)}) to stay under budget")
            s.reset()
            _clear_metal_cache()
            projected -= freed
            evicted += 1
    if projected > MEM_BUDGET_BYTES:
        msg = (f"projected {_gb(projected)} over budget {MEM_BUDGET_GB:g}GB "
               f"after evicting {evicted} slot(s) "
               f"(prefill={n_prefill}t snapshot={snap_to}t, "
               f"~{_gb(int(_BYTES_PER_TOK))}/tok)")
        if MEM_STRICT:
            raise MemoryBudgetExceeded(msg)
        print(f"[qwen-server] governor WARNING (strict off, proceeding): {msg}")
    elif evicted or MEM_LOG:
        print(f"[qwen-server] governor: ok, projected {_gb(projected)} / "
              f"{MEM_BUDGET_GB:g}GB (evicted {evicted}, "
              f"~{_gb(int(_BYTES_PER_TOK))}/tok)")


def _prefill(cache, toks: List[int]):
    """Feed tokens into the cache in chunks, no sampling. This is the manual
    equivalent of generate_step's prefill loop; we need it standalone so the
    snapshot can be taken at an exact point BEFORE generation begins."""
    i = 0
    while i < len(toks):
        chunk = mx.array(toks[i:i + PREFILL_STEP])[None]
        model(chunk, cache=cache)
        arrs = _state_arrays(cache)
        if arrs:
            mx.eval(arrs)
        _clear_metal_cache()
        i += PREFILL_STEP


# --------------------------------------------------------------------------- #
# Repo map — offline structural overview of the WORKSPACE THE REQUEST CAME FROM
#
# Built/looked-up here (filesystem work, no MLX) and injected into the stable
# prefix in build_prompt. Keyed by the absolute workspace path so it follows
# whichever VS Code window is asking, not the server's cwd. The rendered string
# is deterministic for a given tree state, which is what lets the KV snapshot
# cache reuse the injected prefix across turns.
# --------------------------------------------------------------------------- #

# Directories we never descend into (noise / build output / vendored code).
_REPOMAP_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".venv", "venv", "env", "dist", "build",
    "out", ".next", ".nuxt", ".svelte-kit", "target", "bin", "obj", ".idea",
    ".vscode", "coverage", ".cache", ".turbo", ".gradle", "vendor",
    ".terraform", "__snapshots__", ".parcel-cache", ".pnpm-store",
}

# Language-specific top-level symbol patterns. Deliberately lightweight (regex,
# head-of-file) rather than tree-sitter — good enough to answer "where does X
# live" and trivial to run over a large tree.
_JS_PATS = [
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>)", re.M),
]
_TS_PATS = _JS_PATS + [
    re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][\w$]*)\s*[=<]", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)", re.M),
]
_SYMBOL_PATTERNS: Dict[str, List["re.Pattern"]] = {
    ".py": [re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M),
            re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)],
    ".js": _JS_PATS, ".jsx": _JS_PATS, ".mjs": _JS_PATS, ".cjs": _JS_PATS,
    ".ts": _TS_PATS, ".tsx": _TS_PATS,
    ".go": [re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", re.M),
            re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", re.M)],
    ".rs": [re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)", re.M),
            re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.M)],
    ".java": [re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:class|interface|enum)\s+([A-Za-z_]\w*)", re.M)],
    ".cs": [re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+|sealed\s+|abstract\s+|partial\s+)*(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)", re.M)],
    ".rb": [re.compile(r"^\s*(?:class|module)\s+([A-Za-z_]\w*)", re.M),
            re.compile(r"^\s*def\s+([A-Za-z_][\w?!]*)", re.M)],
}

# Files worth listing in the tree even if we don't extract symbols from them.
_REPOMAP_AUX_EXTS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".sql", ".sh",
    ".css", ".scss", ".less", ".html", ".vue", ".svelte", ".proto", ".graphql",
    ".tf", ".env", ".gradle", ".xml",
}
_REPOMAP_SPECIAL_FILES = {
    "Dockerfile", "Makefile", "Procfile", "requirements.txt", "package.json",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "CMakeLists.txt",
}
_REPOMAP_INDEX_EXTS = set(_SYMBOL_PATTERNS) | _REPOMAP_AUX_EXTS

_WS_FOLDERS_RE = re.compile(
    r"working in a workspace with the following folders:\s*\n((?:[ \t]*-[ \t]*.+(?:\n|$))+)"
)

# root path -> {"sig": str|None, "text": str, "built_at": float}
_REPOMAP_CACHE: Dict[str, Dict[str, Any]] = {}


def _extract_workspace_root(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Pull the FIRST workspace folder out of Copilot's <workspace_info> block.
    This is the VS Code window's open folder — the thing we map — and is NOT the
    server's cwd. Returns None for requests with no workspace block (e.g. title
    generation / summary side-requests), which simply get no map."""
    for m in messages:
        c = m.get("content")
        if not isinstance(c, str) or "following folders" not in c:
            continue
        mt = _WS_FOLDERS_RE.search(c)
        if not mt:
            continue
        for line in mt.group(1).splitlines():
            line = line.strip()
            if line.startswith("-"):
                return line[1:].strip()
    return None


def _repo_signature(root: str) -> Optional[str]:
    """Cheap freshness key: the workspace's git HEAD. Changes on commit (so the
    map refreshes when code lands), stable across plain file saves (so it doesn't
    churn the cache mid-session). None when not a git repo -> we fall back to the
    TTL alone."""
    try:
        r = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "git:" + r.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _extract_symbols(path: str, ext: str) -> List[str]:
    pats = _SYMBOL_PATTERNS.get(ext)
    if not pats:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(REPOMAP_MAX_FILE_BYTES)
    except Exception:
        return []
    names: List[str] = []
    seen = set()
    for rx in pats:
        for m in rx.finditer(text):
            n = m.group(1)
            if n and n not in seen:
                seen.add(n)
                names.append(n)
                if len(names) >= REPOMAP_SYMS_PER_FILE:
                    return names
    return names


def _walk_repo(root: str) -> Tuple[List[Tuple[str, List[str]]], bool]:
    """Walk the tree once. Returns (files, truncated) where files is a list of
    (relpath_with_forward_slashes, symbols) and truncated is True if we hit
    REPOMAP_MAX_FILES."""
    ignore = _REPOMAP_IGNORE_DIRS | set(REPOMAP_EXTRA_IGNORES)
    files: List[Tuple[str, List[str]]] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored / hidden dirs in place so os.walk doesn't descend them.
        dirnames[:] = [d for d in dirnames
                       if d not in ignore and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _REPOMAP_INDEX_EXTS and fn not in _REPOMAP_SPECIAL_FILES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            syms = (_extract_symbols(full, ext)
                    if (REPOMAP_SYMBOLS and ext in _SYMBOL_PATTERNS) else [])
            files.append((rel, syms))
            if len(files) >= REPOMAP_MAX_FILES:
                truncated = True
                return files, truncated
    return files, truncated


def _rank_key(item: Tuple[str, List[str]]):
    """Rank for the char budget: real source before tests/config, shallower
    before deeper, symbol-rich before symbol-poor. Lower sorts first / survives
    truncation."""
    rel, syms = item
    low = rel.lower()
    is_test = any(t in low for t in
                  ("test", "spec", "__mocks__", "fixtures", "mock", "/e2e/"))
    return (is_test, rel.count("/"), -len(syms), rel)


def _render_repo_map(root: str, sig: Optional[str]) -> str:
    """Walk + rank + budget + render. The output is deterministic for a given
    tree state (no timestamps, sorted order), so an unchanged repo renders the
    identical string and the KV snapshot prefix stays valid."""
    files, truncated = _walk_repo(root)
    total = len(files)

    # Budget: keep highest-ranked files until we'd blow REPOMAP_MAX_CHARS. We
    # estimate each file's rendered cost (path tail + symbols) before rendering.
    ranked = sorted(files, key=_rank_key)
    kept: List[Tuple[str, List[str]]] = []
    budget = 0
    for rel, syms in ranked:
        cost = len(rel) + sum(len(s) + 2 for s in syms) + 8
        if budget + cost > REPOMAP_MAX_CHARS and kept:
            break
        budget += cost
        kept.append((rel, syms))
    omitted = total - len(kept)

    # Render the kept files as an indented tree (sorted by path for readability).
    tree: Dict[str, Any] = {}
    for rel, syms in sorted(kept):
        parts = rel.split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p + "/", {})
        node[parts[-1]] = syms

    lines: List[str] = []

    def emit(node: Dict[str, Any], depth: int):
        for key in sorted(node):
            val = node[key]
            pad = "  " * depth
            if isinstance(val, dict):
                lines.append(f"{pad}{key}")
                emit(val, depth + 1)
            else:
                tail = (" — " + ", ".join(val)) if val else ""
                lines.append(f"{pad}{key}{tail}")

    emit(tree, 0)

    sig_str = sig if sig else "no-git"
    header = (
        f"<repoMap>\n"
        f"Structural map of the workspace at {root} "
        f"({len(kept)} of {total} files{'+' if truncated else ''} indexed, "
        f"{sig_str}). Paths are relative to the workspace root; "
        f"after each file are its top-level symbols. Use this to locate code "
        f"directly instead of searching, then read/grep only the files you need.\n"
    )
    footer = ""
    if omitted > 0:
        footer = (f"\n... {omitted} lower-priority file(s) omitted to fit budget; "
                  f"use file_search / grep_search to reach them.")
    return header + "\n".join(lines) + footer + "\n</repoMap>"


def _get_repo_map(root: str) -> Optional[str]:
    """Return the cached/rebuilt map text for an absolute workspace root.
    Cache flow: fresh-within-TTL -> return as-is (no walk); else recheck git
    signature -> reuse identical text if unchanged, rebuild otherwise. Falls
    back to disk across restarts. Returns None if the path isn't a real dir on
    this host (e.g. the server runs on a different machine than VS Code)."""
    norm = os.path.normpath(root)
    if not os.path.isdir(norm):
        return None
    # Cache key is case-/sep-normalized so the same workspace doesn't fragment
    # into multiple entries when casing differs (Windows sends "c:\", os.getcwd
    # gives "C:\"). The map text still DISPLAYS `norm` verbatim, which is stable
    # per session because Copilot sends one consistent casing.
    key = os.path.normcase(norm)
    now = time.time()

    ent = _REPOMAP_CACHE.get(key)
    if ent and now - ent["built_at"] < REPOMAP_TTL:
        return ent["text"]                       # fast path: byte-stable, no I/O

    if ent is None:
        ent = _load_repomap_disk(key)            # survive server restarts
        if ent:
            _REPOMAP_CACHE[key] = ent

    sig = _repo_signature(norm)
    if ent and sig is not None and ent.get("sig") == sig:
        ent["built_at"] = now                    # unchanged commit: keep text
        return ent["text"]
    if ent and sig is None and now - ent["built_at"] < REPOMAP_TTL:
        return ent["text"]

    text = _render_repo_map(norm, sig)
    ent = {"sig": sig, "text": text, "built_at": now}
    _REPOMAP_CACHE[key] = ent
    _save_repomap_disk(key, ent)
    print(f"[qwen-server] repo map BUILT for {norm} "
          f"({len(text)} chars, {sig or 'no-git'})")
    return text


def _repomap_disk_path(root: str) -> str:
    h = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
    return os.path.join(REPOMAP_DIR, f"{h}.json")


def _load_repomap_disk(root: str) -> Optional[Dict[str, Any]]:
    path = _repomap_disk_path(root)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("root") == root and isinstance(d.get("text"), str):
            return {"sig": d.get("sig"), "text": d["text"],
                    "built_at": float(d.get("built_at", 0.0))}
    except Exception:
        pass
    return None


def _save_repomap_disk(root: str, ent: Dict[str, Any]):
    try:
        os.makedirs(REPOMAP_DIR, exist_ok=True)
        with open(_repomap_disk_path(root), "w", encoding="utf-8") as f:
            json.dump({"root": root, **ent}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[qwen-server] repo map disk save failed: {e!r}")


def _inject_repo_map(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append a <repoMap> block to the user message that carries <workspace_info>
    (which sits before <context>, hence inside the snapshot-cached stable
    prefix). No-op when disabled, when there's no workspace block, or when the
    workspace path doesn't resolve on this host. Worker-independent: pure Python,
    runs in the request handler before tokenization."""
    if not REPOMAP_ENABLED:
        return messages
    root = _extract_workspace_root(messages)
    if not root:
        return messages
    try:
        block = _get_repo_map(root)
    except Exception as e:
        print(f"[qwen-server] repo map failed for {root!r}: {e!r}")
        return messages
    if not block:
        return messages

    out: List[Dict[str, Any]] = []
    injected = False
    for m in messages:
        c = m.get("content")
        if (not injected and isinstance(c, str) and "following folders" in c):
            m = {**m, "content": c + "\n" + block}
            injected = True
        out.append(m)
    if injected:
        print(f"[qwen-server] repo map injected for {root} ({len(block)} chars)")
    return out


# --------------------------------------------------------------------------- #
# Request (OpenAI messages) -> Qwen chat-template prompt string
# --------------------------------------------------------------------------- #
def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
            parts.append(p.get("text", ""))
        elif isinstance(p, str):
            parts.append(p)
    return "".join(parts)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten list-style content; convert OpenAI tool_call JSON-string arguments
    into dicts (the Qwen template iterates the object, and passing a string trips
    the 'Can only get item pairs from a mapping' template crash). Preserve
    reasoning_content on assistant turns for preserve_thinking."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        msg: Dict[str, Any] = {"role": role}
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                a = fn.get("arguments", "")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except Exception:
                        fn["arguments"] = a
                tcs.append({**tc, "function": fn})
            msg["tool_calls"] = tcs
            msg["content"] = _text_of(m.get("content"))
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
        elif role == "assistant":
            msg["content"] = _text_of(m.get("content"))
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
        elif role == "tool":
            msg["content"] = _text_of(m.get("content"))
            if m.get("name"):
                msg["name"] = m["name"]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
        else:
            msg["content"] = _text_of(m.get("content"))
        out.append(msg)
    return out


def build_prompt(body: Dict[str, Any]) -> str:
    messages = _normalize_messages(body.get("messages", []))
    messages = _inject_repo_map(messages)
    tools = body.get("tools") or None
    if tools and SORT_TOOLS:
        try:
            tools = sorted(
                tools,
                key=lambda t: (t.get("function", t) or {}).get("name") or "",
            )
        except Exception:
            pass

    enable_thinking = body.get("enable_thinking")
    if enable_thinking is None:
        effort = str(body.get("reasoning_effort", "")).lower()
        if effort == "none":
            enable_thinking = False
        elif effort in ("low", "medium", "high"):
            enable_thinking = True
        else:
            enable_thinking = THINKING_DEFAULT

    extra = {"preserve_thinking": True}
    extra.update(dict(body.get("chat_template_kwargs") or {}))

    def render(**kw) -> str:
        return tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw
        )

    for attempt in (
        dict(enable_thinking=enable_thinking, **extra),
        dict(enable_thinking=enable_thinking),
        dict(**extra),
        dict(),
    ):
        try:
            return _apply_pins(render(**attempt))
        except TypeError:
            continue
    return _apply_pins(render())


def _apply_pins(prompt: str) -> str:
    for rx, repl in PIN_PATTERNS:
        prompt = rx.sub(repl, prompt)
    return prompt


def _starts_in_think(prompt: str) -> bool:
    """True if the template already opened a <think> block at the end of the
    prompt, so generation starts mid-reasoning with no opening tag emitted."""
    last_open = prompt.rfind("<think>")
    if last_open == -1:
        return False
    return prompt.find("</think>", last_open) == -1


def _snapshot_boundary(prompt: str, tokens: List[int], start: int) -> int:
    """Token index up to which the cache is snapshotted. Prefer the STRUCTURAL
    marker: snapshot everything before the volatile per-turn block (default
    "<context>"), so the boundary auto-fits however long the tail grows instead
    of guessing a token count. Fall back to the fixed holdback when the marker
    isn't present (e.g. Copilot side-requests with no context block).

    The boundary is clamped to [start, n-1]: never before what we've already
    reused, never the whole prompt (at least one token must remain to generate).
    Re-encoding the prefix can differ from the full tokenization by a token at
    the split, but the marker is newline-preceded so the boundary is stable in
    practice, and a token of slack only means re-prefilling one extra token."""
    n = len(tokens)
    base = n - SNAPSHOT_HOLDBACK
    if SNAPSHOT_MARKER:
        idx = prompt.rfind(SNAPSHOT_MARKER)
        if idx > 0:
            try:
                base = len(tok.encode(prompt[:idx]))
            except Exception:
                pass
    return min(max(start, base), n - 1)


# --------------------------------------------------------------------------- #
# Qwen3-Coder XML tool-call parsing (identical to GGUF server)
# --------------------------------------------------------------------------- #
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


def _strip_wrap_newlines(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _coerce(val: str, schema: Optional[Dict[str, Any]]) -> Any:
    t = (schema or {}).get("type")
    s = val.strip()
    try:
        if t == "integer":
            return int(s)
        if t == "number":
            return float(s)
        if t == "boolean":
            return s.lower() == "true"
        if t in ("object", "array"):
            return json.loads(s)
        if t == "string":
            return _strip_wrap_newlines(val)
        return json.loads(s)
    except Exception:
        return _strip_wrap_newlines(val)


def _props_for(func_name: str, tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not tools:
        return {}
    for tdef in tools:
        fn = tdef.get("function", tdef)
        if fn.get("name") == func_name:
            return (fn.get("parameters") or {}).get("properties", {}) or {}
    return {}


def parse_tool_block(raw: str, tools: Optional[List[Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    raw = raw.strip()
    fm = _FUNC_RE.search(raw)
    if fm:
        name = fm.group(1).strip()
        body = fm.group(2)
        props = _props_for(name, tools)
        args: Dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(body):
            key = pm.group(1).strip()
            args[key] = _coerce(pm.group(2), props.get(key))
        return name, args
    try:
        obj = json.loads(raw)
        name = obj.get("name", "")
        a = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        return name, (a if isinstance(a, dict) else {})
    except Exception:
        return "", {}


# --------------------------------------------------------------------------- #
# Streaming parser: Qwen text stream -> reasoning / content / tool_calls
# --------------------------------------------------------------------------- #
class QwenStreamParser:
    _MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
    _HOLD = max(len(m) for m in _MARKERS) - 1

    def __init__(self, initial_mode: str = "content", tools: Optional[List[Dict[str, Any]]] = None):
        self.buf = ""
        self.mode = initial_mode          # "content" | "reasoning"
        self.in_tool = False
        self._tool_buf = ""
        self.tools = tools
        self.tool_calls: List[Dict[str, Any]] = []

    def _emit(self, text, events):
        if text:
            events.append((self.mode, text))

    def feed(self, delta: str, final: bool = False):
        self.buf += delta
        events: List[tuple] = []
        while True:
            if self.in_tool:
                idx = self.buf.find("</tool_call>")
                if idx == -1:
                    cut = len(self.buf) if final else max(0, len(self.buf) - self._HOLD)
                    self._tool_buf += self.buf[:cut]
                    self.buf = self.buf[cut:]
                    if final:                 # cut off mid-call: finalize anyway
                        self._finalize_tool()
                        self.in_tool = False
                    break
                self._tool_buf += self.buf[:idx]
                self.buf = self.buf[idx + len("</tool_call>"):]
                self._finalize_tool()
                self.in_tool = False
                continue

            cands = ["</think>"] if self.mode == "reasoning" else ["<think>", "<tool_call>"]
            best_idx, best_marker = None, None
            for m in cands:
                i = self.buf.find(m)
                if i != -1 and (best_idx is None or i < best_idx):
                    best_idx, best_marker = i, m

            if best_idx is None:
                cut = len(self.buf) if final else max(0, len(self.buf) - self._HOLD)
                self._emit(self.buf[:cut], events)
                self.buf = self.buf[cut:]
                break

            self._emit(self.buf[:best_idx], events)
            self.buf = self.buf[best_idx + len(best_marker):]
            if best_marker == "<think>":
                self.mode = "reasoning"
            elif best_marker == "</think>":
                self.mode = "content"
            elif best_marker == "<tool_call>":
                self.in_tool = True
        return events

    def _finalize_tool(self):
        raw = self._tool_buf
        self._tool_buf = ""
        name, args = parse_tool_block(raw, self.tools)
        if not name:                          # never emit a nameless tool call
            print(f"[qwen-server] dropping unparseable tool block: {raw[:200]!r}")
            return
        self.tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })

    def finish(self):
        return self.feed("", final=True), self.tool_calls


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _presence_processor(penalty: float, context: int = 512):
    """Additive OpenAI-style presence penalty over recently seen tokens.
    mlx_lm only ships multiplicative repetition_penalty; Qwen3.6's recommended
    anti-loop knob is presence, so implement it as a logits processor."""
    def proc(toks, logits):
        try:
            if toks is None or toks.size == 0:
                return logits
            recent = toks[-context:]
            logits[:, recent] = logits[:, recent] - penalty
        except Exception:
            pass
        return logits
    return proc


# --------------------------------------------------------------------------- #
# Inference: stream text deltas out of MLX (+ perf instrumentation)
# --------------------------------------------------------------------------- #
def run_mlx(prompt: str, body: Dict[str, Any]):
    """Yield generated text deltas. WORKER THREAD ONLY (all Metal work)."""
    def pick(key, default, cast):
        v = body.get(key)
        return default if v is None else cast(v)

    temp = pick("temperature", DEF_TEMP, float)
    top_p = pick("top_p", DEF_TOP_P, float)
    top_k = pick("top_k", DEF_TOP_K, int)
    min_p = pick("min_p", DEF_MIN_P, float)
    repeat = pick("repetition_penalty", DEF_REPEAT, float)
    presence = pick("presence_penalty", DEF_PRESENCE, float)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or DEFAULT_MAX_TOKENS)

    tokens: List[int] = tok.encode(prompt)
    n_prompt = len(tokens)
    if MAX_CONTEXT and n_prompt > MAX_CONTEXT:
        raise MemoryBudgetExceeded(
            f"prompt {n_prompt}t exceeds QWEN_MAX_CONTEXT={MAX_CONTEXT}t")
    slot, start = _acquire_slot(tokens)
    if start == 0 and slot.tokens:
        slot.reset()   # stale snapshot that didn't match: start cold

    # Snapshot point: cache up to the structural marker (everything before the
    # volatile per-turn tail), falling back to the fixed holdback. Keeps the
    # stable prefix cached while date/terminals/editor/request recompute fresh.
    snap_to = _snapshot_boundary(prompt, tokens, start)
    # Governor: make room (evicting LRU slots) or refuse, BEFORE we allocate
    # this request's KV. Must run on the worker thread, which we are.
    _enforce_budget(slot, max(0, snap_to - start), snap_to)
    pct = (100 * start // n_prompt) if n_prompt else 0
    print(f"[qwen-server] slot={slot.idx} prompt={n_prompt}t "
          f"kv-reuse={start}t ({pct}%) prefill={n_prompt - start}t")
    _log_mem("req-start", slot.idx)

    sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
    processors = make_logits_processors(
        repetition_penalty=repeat if repeat and repeat != 1.0 else None,
        repetition_context_size=256,
    ) or []
    if presence:
        processors.append(_presence_processor(presence))

    t0 = time.perf_counter()
    if snap_to > start:
        _a_before = _mx_mem("get_active_memory")
        _prefill(slot.cache, tokens[start:snap_to])
        slot.tokens = tokens[:snap_to]
        # Calibrate the per-token KV estimate from this real allocation delta.
        _update_bytes_per_tok(_a_before, _mx_mem("get_active_memory"),
                              snap_to - start)
    t1 = time.perf_counter()
    slot.last_used = time.time()
    # Generation runs on a COPY; the slot keeps the clean prompt-only state.
    work = _copy_cache(slot.cache)
    if CACHE_DEBUG:
        n_new = snap_to - start
        rate = (n_new / (t1 - t0)) if (n_new and t1 > t0) else 0
        print(f"[qwen-server] slot={slot.idx} snapshot={len(slot.tokens)}t"
              + (f" prefill {n_new}t @ ~{rate:.0f} t/s" if n_new else "")
              + f", copy {time.perf_counter() - t1:.2f}s")
    # Peak after prefill + the working-copy clone: this is the transient
    # high-water mark (snapshot KV + its live copy both resident) that most
    # often tips a 128k context over the 64GB ceiling.
    _log_mem("after-prefill+copy", slot.idx)
    suffix = tokens[snap_to:]

    t_first = None
    n_out = 0
    for resp in stream_generate(
        model, tok, prompt=suffix,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=processors,
        prompt_cache=work,
        prefill_step_size=PREFILL_STEP,
    ):
        piece = resp.text
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
                dt = t_first - t0
                print(f"[qwen-server] TTFT={dt:.2f}s")
            n_out += 1
            yield piece

    t_end = time.perf_counter()
    if t_first is not None and n_out > 1 and t_end > t_first:
        print(f"[qwen-server] decode: {n_out} chunks in {t_end - t_first:.2f}s "
              f"(~{n_out / (t_end - t_first):.1f} t/s)")
    # Drop the working copy and return its buffers to the allocator — at
    # long contexts this is gigabytes that would otherwise sit in MLX's
    # buffer pool between requests.
    del work
    _clear_metal_cache()
    _log_mem("after-cleanup", slot.idx)


# --------------------------------------------------------------------------- #
# Chat logging (visibility): one JSON per request in ./chats
# --------------------------------------------------------------------------- #
def _log_chat(cid: str, created: int, body: Dict[str, Any], prompt: str,
              reasoning: str, content: str, tool_calls: List[Dict[str, Any]],
              stream: bool, duration_s: float):
    if not CHAT_LOG:
        return
    try:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)
        record = {
            "id": cid,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created)),
            "duration_s": round(duration_s, 2),
            "stream": stream,
            "model": MODEL_ID,
            "params": {k: body.get(k) for k in (
                "temperature", "top_p", "top_k", "min_p", "max_tokens",
                "max_completion_tokens", "presence_penalty", "frequency_penalty",
                "repetition_penalty", "reasoning_effort", "enable_thinking",
            ) if body.get(k) is not None},
            "tools": [(t.get("function", t) or {}).get("name")
                      for t in (body.get("tools") or [])],
            "messages": body.get("messages"),
            "rendered_prompt": prompt,
            "response": {
                "reasoning_content": reasoning or None,
                "content": content or None,
                "tool_calls": tool_calls or None,
            },
        }
        fname = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(created))}-{cid[-8:]}.json"
        path = os.path.join(CHAT_LOG_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[qwen-server] chat log -> {path}")
    except Exception as e:
        print(f"[qwen-server] chat log failed: {e!r}")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = bool(body.get("stream", False))
    tools = body.get("tools") or None
    created = int(time.time())
    t_req = time.perf_counter()
    cid = "chatcmpl-" + uuid.uuid4().hex

    prompt = build_prompt(body)
    init_mode = "reasoning" if _starts_in_think(prompt) else "content"

    if not stream:
        def generate_all():
            parser = QwenStreamParser(initial_mode=init_mode, tools=tools)
            reasoning, content = "", ""
            for delta in run_mlx(prompt, body):
                for kind, txt in parser.feed(delta):
                    if kind == "reasoning":
                        reasoning += txt
                    else:
                        content += txt
            tail, tool_calls = parser.finish()
            for kind, txt in tail:
                if kind == "reasoning":
                    reasoning += txt
                else:
                    content += txt
            return reasoning, content, tool_calls

        try:
            reasoning, content, tool_calls = await asyncio.wrap_future(GPU.submit(generate_all))
        except MemoryBudgetExceeded as e:
            print(f"[qwen-server] refused (memory budget): {e}")
            return JSONResponse(status_code=503, content={"error": {
                "message": str(e), "type": "memory_budget_exceeded",
                "code": "context_too_large"}})
        _log_chat(cid, created, body, prompt, reasoning, content, tool_calls,
                  stream=False, duration_s=time.perf_counter() - t_req)
        message: Dict[str, Any] = {"role": "assistant",
                                   "content": content if content else None}
        if reasoning.strip():
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        })

    # Streaming: producer runs generation on the worker thread, pushing text
    # deltas to a queue; the SSE generator only drains the queue and parses.
    # No MLX call ever leaves the worker thread.
    def sse():
        parser = QwenStreamParser(initial_mode=init_mode, tools=tools)
        q: "queue.Queue" = queue.Queue(maxsize=256)
        acc_reasoning, acc_content = [], []

        def produce():
            try:
                for delta in run_mlx(prompt, body):
                    q.put(("txt", delta))
            except Exception as e:
                q.put(("err", f"{type(e).__name__}: {e}"))
            finally:
                q.put(("end", None))

        GPU.submit(produce)

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
            return "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }) + "\n\n"

        yield chunk({"role": "assistant"})

        errored = False
        err_msg = ""
        while True:
            kind, val = q.get()
            if kind == "end":
                break
            if kind == "err":
                print(f"[qwen-server] generation error: {val}")
                errored = True
                err_msg = val
                break
            for ev_kind, txt in parser.feed(val):
                (acc_reasoning if ev_kind == "reasoning" else acc_content).append(txt)
                yield chunk({"reasoning_content": txt} if ev_kind == "reasoning"
                            else {"content": txt})

        tail, tool_calls = parser.finish()
        for ev_kind, txt in tail:
            (acc_reasoning if ev_kind == "reasoning" else acc_content).append(txt)
            yield chunk({"reasoning_content": txt} if ev_kind == "reasoning"
                        else {"content": txt})

        if errored:
            if err_msg:
                yield chunk({"content": f"\n[server: {err_msg}]"})
            yield chunk({}, finish="stop")
        elif tool_calls:
            tc_delta = [{"index": i, **tc} for i, tc in enumerate(tool_calls)]
            yield chunk({"tool_calls": tc_delta})       # tool calls (finish_reason null)
            yield chunk({}, finish="tool_calls")        # separate terminal chunk
        else:
            yield chunk({}, finish="stop")

        _log_chat(cid, created, body, prompt,
                  "".join(acc_reasoning), "".join(acc_content), tool_calls,
                  stream=True, duration_s=time.perf_counter() - t_req)
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)