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

  TWO PRODUCTS from one cached per-root index:
    - STATIC <repoMap>: role-ranked detail (Angular/.NET conventions + PageRank
      reference centrality) PLUS a collapsed directory breadth summary, injected
      before <context> -> snapshot-cached stable prefix, paid once per repo.
    - DYNAMIC <relevantFiles>: deterministic retrieval for the CURRENT question
      (no model, no swap — microseconds off the prebuilt index), injected before
      <userRequest> -> volatile tail, small and query-targeted. This is the
      "agent in the middle" that points the model at the right files turn one.
  MULTI-ROOT: every workspace folder is indexed (your Angular + .NET repos in
  one window both get mapped, not just the first).

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
  QWEN_REPOMAP_MAX_CHARS (static detail budget), QWEN_REPOMAP_BREADTH[_MAX_CHARS],
  QWEN_REPOMAP_GRAPH[_BOOST/_MAX_FILES], QWEN_REPOMAP_HINT[_FILES],
  QWEN_REPOMAP_SYMBOLS, QWEN_REPOMAP_TREE_SITTER, QWEN_REPOMAP_MAX_FILES,
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
    save_prompt_cache,
    load_prompt_cache,
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

# --- Disk-backed snapshot cache ------------------------------------------- #
# Spill idle conversation snapshots to SSD instead of discarding them when a
# slot is recycled for a different conversation, and reload them on return.
# Switching back to an old chat then costs a disk read + a tiny suffix prefill
# instead of a full ~30k-token reprefill. RAM still holds only the live working
# set (CACHE_SLOTS resident slots); the heavy KV lives on disk, and only a small
# token-ledger index stays in memory for prefix matching (~4 B/token).
#
# DEFAULT OFF. Correctness depends on save_prompt_cache/load_prompt_cache
# reproducing THIS model's quantized-KV + recurrent (hybrid-attention) cache
# state exactly; a bad round-trip corrupts context SILENTLY (garbage output, not
# a cache miss). Run test_cache_roundtrip.py on the target machine first, then
# set QWEN_DISK_CACHE=1. Single-model deployment -> no model/version
# invalidation needed; a stale ledger simply fails the exact-prefix test and is
# ignored.
DISK_CACHE = os.environ.get("QWEN_DISK_CACHE", "0") == "1"
DISK_CACHE_DIR = os.environ.get("QWEN_DISK_CACHE_DIR",
                                os.path.join(os.getcwd(), "cachesnaps"))
# Total disk budget for spilled snapshots (GB); oldest evicted past it. 0 = off.
DISK_CACHE_MAX_GB = float(os.environ.get("QWEN_DISK_CACHE_MAX_GB", "100"))
DISK_CACHE_MAX_BYTES = int(DISK_CACHE_MAX_GB * (1 << 30))
# Don't spill snapshots shorter than this — reprefilling them is already cheap,
# and the disk write + reload overhead wouldn't pay off.
DISK_CACHE_MIN_TOKENS = int(os.environ.get("QWEN_DISK_CACHE_MIN_TOKENS", "2048"))

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
# Master switch. When on, every request gets:
#   (1) a STATIC <repoMap> (role-ranked detail + collapsed directory breadth)
#       injected before <context> -> snapshot-cached stable prefix, paid once;
#   (2) a DYNAMIC <relevantFiles> hint (deterministic retrieval for THIS
#       question) injected before <userRequest> -> volatile tail, cheap+targeted.
# Multi-root workspaces are fully handled: every workspace folder is indexed.
REPOMAP_ENABLED = os.environ.get("QWEN_REPOMAP", "1") == "1"

# Where built indexes are cached on disk (survives restarts). One JSON per
# workspace root, named by a hash of the absolute path.
REPOMAP_DIR = os.environ.get("QWEN_REPOMAP_DIR", os.path.join(os.getcwd(), "repomaps"))

# How long a built index is trusted before we re-check the repo signature (git
# HEAD) / rebuild. Within this window we serve from memory WITHOUT walking the
# tree, so the injected static prefix is byte-stable and the KV snapshot stays
# valid no matter how many edits you make mid-session. Seconds.
REPOMAP_TTL = float(os.environ.get("QWEN_REPOMAP_TTL", "300"))

# Char budget for the STATIC detailed section (per root). It's snapshot-cached,
# so a generous budget is nearly free after turn 1; raise it on big repos.
REPOMAP_MAX_CHARS = int(os.environ.get("QWEN_REPOMAP_MAX_CHARS", "16000"))

# Stop walking after this many indexable files (guards a pathological tree).
REPOMAP_MAX_FILES = int(os.environ.get("QWEN_REPOMAP_MAX_FILES", "8000"))

# Per-file byte cap for symbol extraction (we only scan the head of each file).
REPOMAP_MAX_FILE_BYTES = int(os.environ.get("QWEN_REPOMAP_MAX_FILE_BYTES", "200000"))

# Extract top-level symbols (classes/services/components/controllers/...) per
# file. Symbols are what make the map answer "where does X live".
REPOMAP_SYMBOLS = os.environ.get("QWEN_REPOMAP_SYMBOLS", "1") == "1"

# Max symbols listed per file (keeps fat files from dominating the budget).
REPOMAP_SYMS_PER_FILE = int(os.environ.get("QWEN_REPOMAP_SYMS_PER_FILE", "12"))

# Collapsed directory breadth summary for the files NOT shown in detail, so the
# model sees the whole-repo shape (and won't assume code is missing) for cheap.
REPOMAP_BREADTH = os.environ.get("QWEN_REPOMAP_BREADTH", "1") == "1"
REPOMAP_BREADTH_MAX_CHARS = int(os.environ.get("QWEN_REPOMAP_BREADTH_MAX_CHARS", "8000"))

# Reference-graph centrality: build a symbol-reference graph (file A mentions a
# type/class/component defined in file B -> edge) and PageRank it. Central files
# (imported/used everywhere) get a ranking boost on top of their role weight.
# Works for BOTH stacks (it keys on symbol names, not import/namespace syntax).
REPOMAP_GRAPH = os.environ.get("QWEN_REPOMAP_GRAPH", "1") == "1"
REPOMAP_GRAPH_BOOST = float(os.environ.get("QWEN_REPOMAP_GRAPH_BOOST", "2.0"))
# Skip the graph above this file count (it gets slow); fall back to role weight.
REPOMAP_GRAPH_MAX_FILES = int(os.environ.get("QWEN_REPOMAP_GRAPH_MAX_FILES", "6000"))
# Bytes scanned per file when building reference edges (smaller = faster build).
REPOMAP_GRAPH_READ_BYTES = int(os.environ.get("QWEN_REPOMAP_GRAPH_READ_BYTES", "65536"))

# Dynamic per-question retrieval hint (the "agent in the middle", but
# deterministic: no model, no swap, runs in microseconds from the prebuilt
# index). Picks the files most relevant to THIS request and injects them in the
# volatile tail. This is what replaces the 100k-token investigation.
REPOMAP_HINT = os.environ.get("QWEN_REPOMAP_HINT", "1") == "1"
REPOMAP_HINT_FILES = int(os.environ.get("QWEN_REPOMAP_HINT_FILES", "12"))

# Optional tree-sitter symbol backend (EXPERIMENTAL, default off, regex is the
# tested default). If enabled and tree_sitter_languages is importable, symbols
# are extracted from real parse trees; any failure falls back to regex per file.
#   pip install tree_sitter_languages
REPOMAP_TREE_SITTER = os.environ.get("QWEN_REPOMAP_TREE_SITTER", "0") == "1"

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
          f"static={REPOMAP_MAX_CHARS}c +breadth {REPOMAP_BREADTH_MAX_CHARS}c, "
          f"graph={'on' if REPOMAP_GRAPH else 'off'}, "
          f"hint={'on x%d' % REPOMAP_HINT_FILES if REPOMAP_HINT else 'off'}, "
          f"tree-sitter={'on' if REPOMAP_TREE_SITTER else 'off'}) — "
          f"multi-root, static map -> cached prefix, hint -> volatile tail")
else:
    print("[qwen-server] repo map: OFF (set QWEN_REPOMAP=1 to enable)")
if DISK_CACHE:
    print(f"[qwen-server] disk cache: ON (dir={DISK_CACHE_DIR}, "
          f"cap={('%gGB' % DISK_CACHE_MAX_GB) if DISK_CACHE_MAX_BYTES else 'none'}, "
          f"min={DISK_CACHE_MIN_TOKENS}t) — idle snapshots spill to SSD, "
          f"reload on return")
else:
    print("[qwen-server] disk cache: OFF (set QWEN_DISK_CACHE=1 after "
          "test_cache_roundtrip.py passes)")

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
    (no rollback), so it counts as zero. When QWEN_DISK_CACHE is on, also
    consider disk-spilled snapshots: if one is a longer exact prefix than any
    resident slot, recycle a slot (spilling its current snapshot to disk first)
    and reload the disk snapshot into it. Worker thread only."""
    _disk_ensure_ready()
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

    # Disk-spilled snapshots: reload only when STRICTLY better than the best
    # resident match — a disk read + eval is cheap vs reprefill, but not free.
    if DISK_CACHE:
        disk_entry, disk_len = _disk_find(tokens)
        if disk_entry is not None and disk_len > best_usable:
            slot = _recycle_slot(exclude=best)
            if _disk_load_into(slot, disk_entry):
                print(f"[qwen-server] slot={slot.idx} warm-loaded {disk_len}t "
                      f"from disk (key={disk_entry['key'][:8]})")
                return slot, disk_len
            # load failed: fall through to the RAM/cold path

    if best is not None:
        return best, best_usable
    if CACHE_DEBUG and best_partial is not None:
        _log_divergence(best_partial, tokens, best_partial_k)
    return _recycle_slot(exclude=None), 0


def _recycle_slot(exclude: Optional["_Slot"]) -> "_Slot":
    """Return a slot ready for a fresh (cold or disk-loaded) snapshot. Prefer a
    not-yet-created slot; else evict the LRU slot — spilling its snapshot to disk
    first (QWEN_DISK_CACHE) so switching back to that conversation later is a
    reload, not a reprefill. Worker thread only."""
    if len(_SLOTS) < CACHE_SLOTS:
        s = _Slot(len(_SLOTS))
        _SLOTS.append(s)
        return s
    victim = min((s for s in _SLOTS if s is not exclude),
                 key=lambda x: x.last_used, default=None)
    if victim is None:                       # only slot is `exclude`
        victim = min(_SLOTS, key=lambda x: x.last_used)
    if DISK_CACHE and victim.tokens:
        _disk_save_slot(victim)
    victim.reset()
    return victim


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
# Disk-backed snapshot store — spill idle conversation caches to SSD
#
# RAM holds only the live working set (CACHE_SLOTS slots). When a slot is
# recycled for a DIFFERENT conversation, its snapshot is serialized to disk
# (save_prompt_cache) instead of discarded, keyed by a hash of its exact token
# ledger. A later request whose prompt is an exact prefix-superset of that
# ledger reloads it (load_prompt_cache, safetensors mmap -> bound by SSD
# bandwidth, not reprefill compute) and prefills only the new suffix.
#
# The ledger index lives in RAM (one int list per saved snapshot); the heavy KV
# stays on disk until a request needs it, plus a small sidecar JSON per snapshot
# so the index survives restarts. All access is from the single GPU worker
# thread (serialized) -> no locking.
# --------------------------------------------------------------------------- #

# RAM index: list of {"tokens": List[int], "key": str, "nbytes": int,
#                     "last_used": float}. Heavy KV is the matching .safetensors.
_DISK_INDEX: List[Dict[str, Any]] = []
_DISK_READY = False


def _disk_paths(key: str) -> Tuple[str, str]:
    base = os.path.join(DISK_CACHE_DIR, key)
    return base + ".safetensors", base + ".json"


def _disk_ensure_ready():
    """Rebuild the RAM ledger index from sidecar JSONs on first use (survives
    server restarts). Reads only the small sidecars, never the heavy caches."""
    global _DISK_READY
    if _DISK_READY:
        return
    _DISK_READY = True
    if not DISK_CACHE:
        return
    try:
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        for fn in os.listdir(DISK_CACHE_DIR):
            if not fn.endswith(".json"):
                continue
            key = fn[:-5]
            st_path, js_path = _disk_paths(key)
            if not os.path.exists(st_path):
                continue
            try:
                with open(js_path, encoding="utf-8") as f:
                    d = json.load(f)
                toks = d.get("tokens")
                if not toks:
                    continue
                _DISK_INDEX.append({
                    "tokens": toks, "key": key,
                    "nbytes": int(d.get("nbytes", os.path.getsize(st_path))),
                    "last_used": float(d.get("last_used", 0.0)),
                })
            except Exception:
                continue
        if _DISK_INDEX:
            tot = sum(e["nbytes"] for e in _DISK_INDEX)
            print(f"[qwen-server] disk cache: loaded {len(_DISK_INDEX)} "
                  f"snapshot(s) from {DISK_CACHE_DIR} (~{_gb(tot)})")
    except Exception as e:
        print(f"[qwen-server] disk cache bootstrap failed: {e!r}")


def _disk_key(tokens: List[int]) -> str:
    h = hashlib.sha256()
    h.update(str(len(tokens)).encode())
    h.update(b"\0")
    h.update(",".join(map(str, tokens)).encode())
    return h.hexdigest()[:32]


def _disk_find(tokens: List[int]) -> Tuple[Optional[Dict[str, Any]], int]:
    """Longest disk snapshot whose ledger is an EXACT prefix of `tokens` (and
    strictly shorter, so at least one token remains to prefill/generate)."""
    best, best_len = None, 0
    n = len(tokens)
    for e in _DISK_INDEX:
        led = e["tokens"]
        m = len(led)
        if m <= best_len or m >= n:
            continue
        if led == tokens[:m]:
            best, best_len = e, m
    return best, best_len


def _disk_save_slot(slot: "_Slot"):
    """Serialize a slot's current snapshot to disk (skip tiny/dup). Captures the
    slot at its deepest cached boundary (slot.tokens), so reload gives deep reuse
    and a minimal suffix prefill. Worker thread only."""
    if not DISK_CACHE:
        return
    toks = slot.tokens
    if len(toks) < DISK_CACHE_MIN_TOKENS:
        return
    key = _disk_key(toks)
    for e in _DISK_INDEX:
        if e["key"] == key:
            e["last_used"] = time.time()     # already on disk; just touch
            return
    st_path, js_path = _disk_paths(key)
    try:
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        save_prompt_cache(st_path, slot.cache,
                          metadata={"n": str(len(toks)), "model": MODEL_ID})
        nbytes = os.path.getsize(st_path)
        ent = {"tokens": list(toks), "key": key, "nbytes": nbytes,
               "last_used": time.time()}
        with open(js_path, "w", encoding="utf-8") as f:
            json.dump({"tokens": ent["tokens"], "nbytes": nbytes,
                       "last_used": ent["last_used"], "model": MODEL_ID}, f)
        _DISK_INDEX.append(ent)
        print(f"[qwen-server] disk cache: spilled slot={slot.idx} "
              f"{len(toks)}t ({_gb(nbytes)}) key={key[:8]}")
        _disk_enforce_cap()
    except Exception as e:
        print(f"[qwen-server] disk cache save failed: {e!r}")


def _disk_load_into(slot: "_Slot", entry: Dict[str, Any]) -> bool:
    """Load a disk snapshot into `slot` (cache + token ledger). mmap-backed:
    arrays page in lazily and are materialized by the mx.eval. Returns True on
    success, False (and leaves the slot reset) on any load error."""
    st_path, _ = _disk_paths(entry["key"])
    try:
        cache = load_prompt_cache(st_path)
        slot.cache = cache
        slot.tokens = list(entry["tokens"])
        arrs = _state_arrays(slot.cache)
        if arrs:
            mx.eval(arrs)
        entry["last_used"] = time.time()
        return True
    except Exception as e:
        print(f"[qwen-server] disk cache load failed ({entry['key'][:8]}): {e!r}")
        slot.reset()
        return False


def _disk_enforce_cap():
    """Evict oldest snapshots (safetensors + sidecar + index entry) until total
    bytes fall under QWEN_DISK_CACHE_MAX_GB. No-op when the cap is 0."""
    if not DISK_CACHE_MAX_BYTES:
        return
    tot = sum(e["nbytes"] for e in _DISK_INDEX)
    if tot <= DISK_CACHE_MAX_BYTES:
        return
    for e in sorted(_DISK_INDEX, key=lambda x: x["last_used"]):
        if tot <= DISK_CACHE_MAX_BYTES:
            break
        st_path, js_path = _disk_paths(e["key"])
        for p in (st_path, js_path):
            try:
                os.remove(p)
            except Exception:
                pass
        tot -= e["nbytes"]
        _DISK_INDEX.remove(e)
        print(f"[qwen-server] disk cache: evicted {len(e['tokens'])}t "
              f"({_gb(e['nbytes'])}) key={e['key'][:8]}")


# --------------------------------------------------------------------------- #
# Repo map — offline index of the WORKSPACE(S) THE REQUEST CAME FROM
#
# One cached per-root index drives two products:
#   STATIC  <repoMap>       role-ranked detail + collapsed directory breadth,
#                           injected before <context> (snapshot-cached prefix,
#                           paid once per repo).
#   DYNAMIC <relevantFiles> deterministic retrieval for the CURRENT question,
#                           injected before <userRequest> (volatile tail, cheap
#                           and query-targeted) — the "agent in the middle",
#                           minus the model/swap: it runs in microseconds off the
#                           prebuilt index.
# Keyed by absolute workspace path (follows the asking VS Code window, not the
# server cwd) and validated by git HEAD. Multi-root workspaces index EVERY
# folder. Rendering is deterministic so the cached static prefix stays stable.
# --------------------------------------------------------------------------- #

# Directories we never descend into (noise / build output / vendored code).
_REPOMAP_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".venv", "venv", "env", "dist", "build",
    "out", ".next", ".nuxt", ".svelte-kit", "target", "bin", "obj", ".idea",
    ".vscode", "coverage", ".cache", ".turbo", ".gradle", "vendor",
    ".terraform", "__snapshots__", ".parcel-cache", ".pnpm-store", "Debug",
    "Release", "TestResults", "packages",
}

# Language-specific top-level symbol patterns (regex, head-of-file — the tested
# default backend; tree-sitter is an optional override, see _ts_symbols).
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
_CS_PATS = [
    re.compile(r"^\s*(?:\[[^\]]*\]\s*)*(?:(?:public|private|protected|internal|sealed|abstract|static|partial)\s+)*(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)", re.M),
    re.compile(r"^\s*(?:public|protected|internal)\s+(?:static\s+|virtual\s+|override\s+|async\s+)*[\w<>\[\],\.\?]+\s+([A-Za-z_]\w*)\s*\(", re.M),
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
    ".cs": _CS_PATS,
    ".rb": [re.compile(r"^\s*(?:class|module)\s+([A-Za-z_]\w*)", re.M),
            re.compile(r"^\s*def\s+([A-Za-z_][\w?!]*)", re.M)],
}

# "Major" (graph-node) symbols: type-like DEFINITIONS only. The reference graph
# links a file to every file whose major symbol it mentions — that works for
# both Angular (class/interface) and C# (class/interface/record), keying on
# names rather than import/namespace syntax.
_MAJOR_TS = [
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][\w$]*)\s*[=<]", re.M),
    re.compile(r"^\s*(?:export\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)", re.M),
]
_MAJOR_PATTERNS: Dict[str, List["re.Pattern"]] = {
    ".ts": _MAJOR_TS, ".tsx": _MAJOR_TS,
    ".js": _MAJOR_TS[:1], ".jsx": _MAJOR_TS[:1], ".mjs": _MAJOR_TS[:1], ".cjs": _MAJOR_TS[:1],
    ".py": [re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)],
    ".cs": [re.compile(r"^\s*(?:\[[^\]]*\]\s*)*(?:(?:public|private|protected|internal|sealed|abstract|static|partial)\s+)*(?:class|interface|struct|enum|record)\s+([A-Za-z_]\w*)", re.M)],
    ".go": [re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", re.M)],
    ".rs": [re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.M)],
    ".java": [re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:class|interface|enum)\s+([A-Za-z_]\w*)", re.M)],
    ".rb": [re.compile(r"^\s*(?:class|module)\s+([A-Za-z_]\w*)", re.M)],
}

# Stack-specific enrichment (decorators / namespace / endpoints) layered on top
# of whatever symbol backend ran — this is what makes the map read like the
# architecture rather than a list of class names.
_NG_SELECTOR_RE = re.compile(r"@Component\s*\([^)]*?selector\s*:\s*['\"]([^'\"]+)['\"]", re.S)
_NG_DECORATORS = [
    ("@Component", re.compile(r"@Component\s*\(")),
    ("@Injectable", re.compile(r"@Injectable\s*\(")),
    ("@NgModule", re.compile(r"@NgModule\s*\(")),
    ("@Directive", re.compile(r"@Directive\s*\(")),
    ("@Pipe", re.compile(r"@Pipe\s*\(")),
]
_CS_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][\w.]*)", re.M)
_CS_HTTP_RE = re.compile(r"\[Http(Get|Post|Put|Delete|Patch)(?:\(\s*['\"]([^'\"]*)['\"])?")

# Files worth listing even when we don't extract symbols from them.
_REPOMAP_AUX_EXTS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".sql", ".sh",
    ".css", ".scss", ".less", ".html", ".vue", ".svelte", ".proto", ".graphql",
    ".tf", ".env", ".gradle", ".xml", ".csproj", ".sln", ".razor",
}
_REPOMAP_SPECIAL_FILES = {
    "Dockerfile", "Makefile", "Procfile", "requirements.txt", "package.json",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "CMakeLists.txt",
    "angular.json", "nx.json", "tsconfig.json",
}
_REPOMAP_INDEX_EXTS = set(_SYMBOL_PATTERNS) | _REPOMAP_AUX_EXTS

_WS_FOLDERS_RE = re.compile(
    r"working in a workspace with the following folders:\s*\n((?:[ \t]*-[ \t]*.+(?:\n|$))+)"
)
_USERREQ_RE = re.compile(r"<userRequest>(.*?)</userRequest>", re.S)
_SUBTOK_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Za-z][a-z0-9]*|[0-9]+")
_IDENT_RE = re.compile(r"[A-Za-z_]\w+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "how", "what", "where", "why", "do", "does", "i", "we", "my", "our", "add",
    "fix", "change", "update", "make", "need", "want", "please", "can", "with",
    "this", "that", "it", "use", "using", "when", "get", "set", "new", "file",
    "code", "function", "class", "method", "should", "would", "could", "into",
}

# key (normcased path) -> {"sig", "built_at", "index", "static_text"}
_REPOMAP_CACHE: Dict[str, Dict[str, Any]] = {}


def _read_head(path: str, n: int = REPOMAP_MAX_FILE_BYTES) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(n)
    except Exception:
        return ""


def _subtokens(s: str) -> set:
    """Split an identifier/path into lowercase sub-words (camelCase, kebab, dots,
    slashes) so 'AuthService' and 'auth.guard.ts' both match a query for 'auth'."""
    out = set()
    for part in re.split(r"[^A-Za-z0-9]+", s):
        for w in _SUBTOK_RE.findall(part):
            if len(w) > 1:
                out.add(w.lower())
    return out


def _workspace_roots(messages: List[Dict[str, Any]]) -> List[str]:
    """ALL workspace folders from Copilot's <workspace_info> block (multi-root
    workspaces list several). Empty for side-requests with no workspace block."""
    for m in messages:
        c = m.get("content")
        if not isinstance(c, str) or "following folders" not in c:
            continue
        mt = _WS_FOLDERS_RE.search(c)
        if not mt:
            continue
        roots = []
        for line in mt.group(1).splitlines():
            line = line.strip()
            if line.startswith("-"):
                p = line[1:].strip()
                if p:
                    roots.append(p)
        if roots:
            return roots
    return []


def _extract_question(messages: List[Dict[str, Any]]) -> str:
    """The user's actual request: prefer the last <userRequest> block, else the
    last user message text."""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            mt = _USERREQ_RE.search(m["content"])
            if mt:
                return mt.group(1).strip()
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    return ""


def _repo_signature(root: str) -> Optional[str]:
    """Cheap freshness key: the workspace's git HEAD. Changes on commit (map
    refreshes when code lands), stable across plain saves (no mid-session churn).
    None when not a git repo -> we fall back to the TTL alone."""
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


def _classify(rel: str) -> Tuple[str, float, bool]:
    """Map a path to (role, weight, detail_eligible) using Angular/.NET naming
    conventions. Higher weight => more architecturally important. detail_eligible
    False means it's excluded from the detailed section (too noisy) but still
    counted in the directory breadth."""
    name = rel.rsplit("/", 1)[-1]
    nlow = name.lower()
    low = rel.lower()
    ext = os.path.splitext(name)[1].lower()

    # Excluded from detail (still counted in breadth).
    if nlow.endswith((".spec.ts", ".test.ts")) or ".spec." in nlow or nlow.endswith(("tests.cs", ".test.cs")):
        return ("test", 0.5, False)
    if nlow.endswith((".designer.cs", ".g.cs", ".generated.cs", ".g.ts")):
        return ("generated", 0.2, False)
    if "/migrations/" in low or low.startswith("migrations/"):
        return ("migration", 0.4, False)
    if ext in (".html", ".htm"):
        return ("template", 1.0, False)
    if ext in (".scss", ".css", ".less", ".sass"):
        return ("style", 1.0, False)

    # Angular.
    if nlow.endswith("-routing.module.ts") or nlow in ("app.routes.ts", "app-routing.module.ts"):
        return ("ng-routing", 10.0, True)
    if nlow.endswith(".module.ts"):
        return ("ng-module", 9.0, True)
    if nlow.endswith((".service.ts", ".guard.ts", ".interceptor.ts", ".resolver.ts",
                      ".facade.ts", ".store.ts", ".effects.ts", ".state.ts", ".reducer.ts")):
        return ("ng-service", 9.0, True)
    if nlow.endswith(".component.ts"):
        return ("ng-component", 7.0, True)
    if nlow.endswith((".directive.ts", ".pipe.ts")):
        return ("ng-other", 6.0, True)
    if nlow.endswith((".model.ts", ".dto.ts", ".types.ts", ".interface.ts", ".enum.ts")):
        return ("ng-model", 5.0, True)
    if nlow == "index.ts":
        return ("barrel", 5.0, True)

    # C# / .NET.
    if nlow in ("program.cs", "startup.cs"):
        return ("net-entry", 10.0, True)
    if nlow.endswith("controller.cs"):
        return ("net-controller", 9.0, True)
    if nlow.endswith(("service.cs", "repository.cs", "handler.cs", "manager.cs",
                      "provider.cs", "factory.cs")):
        return ("net-service", 8.0, True)
    if name[:1] == "I" and name[1:2].isupper() and ext == ".cs":
        return ("net-interface", 7.0, True)
    if nlow.endswith(("dto.cs", "model.cs", "entity.cs", "request.cs", "response.cs",
                      "dao.cs", "vm.cs", "viewmodel.cs")):
        return ("net-model", 5.0, True)

    # Build / config / docs.
    if ext in (".csproj", ".sln") or name in _REPOMAP_SPECIAL_FILES:
        return ("build", 7.0, True)
    if ext in _SYMBOL_PATTERNS:
        return ("code", 4.0, True)
    if ext == ".md":
        return ("doc", 2.0, True)
    return ("other", 2.0, ext in _SYMBOL_PATTERNS)


# Optional tree-sitter backend (experimental). Node-type sets are deliberately
# broad; any failure returns None and we fall back to regex.
_TS_LANG = {".ts": "typescript", ".tsx": "tsx", ".js": "javascript",
            ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".py": "python", ".cs": "c_sharp", ".go": "go", ".rs": "rust",
            ".java": "java", ".rb": "ruby"}
_TS_DEF_NODES = {"class_declaration", "interface_declaration", "type_alias_declaration",
                 "enum_declaration", "function_declaration", "method_definition",
                 "method_declaration", "struct_declaration", "record_declaration",
                 "function_definition", "class_definition"}
_TS_MAJOR_NODES = {"class_declaration", "interface_declaration", "type_alias_declaration",
                   "enum_declaration", "struct_declaration", "record_declaration",
                   "class_definition"}
_TS_PARSERS: Dict[str, Any] = {}


def _ts_symbols(text: str, ext: str):
    lang = _TS_LANG.get(ext)
    if not lang:
        return None
    try:
        from tree_sitter_languages import get_parser
        p = _TS_PARSERS.get(lang)
        if p is None:
            p = get_parser(lang)
            _TS_PARSERS[lang] = p
        tree = p.parse(text.encode("utf-8", "ignore"))
        syms: List[str] = []
        major = set()
        seen = set()
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in _TS_DEF_NODES:
                nm = node.child_by_field_name("name")
                if nm is not None:
                    name = nm.text.decode("utf-8", "ignore")
                    if name and name not in seen:
                        seen.add(name)
                        syms.append(name)
                        if node.type in _TS_MAJOR_NODES:
                            major.add(name)
            stack.extend(node.children)
        return syms, major
    except Exception:
        return None


def _regex_symbols(text: str, ext: str) -> Tuple[List[str], set]:
    syms: List[str] = []
    seen = set()
    for rx in _SYMBOL_PATTERNS.get(ext, []):
        for m in rx.finditer(text):
            n = m.group(1)
            if n and n not in seen:
                seen.add(n)
                syms.append(n)
    major = set()
    for rx in _MAJOR_PATTERNS.get(ext, []):
        for m in rx.finditer(text):
            if m.group(1):
                major.add(m.group(1))
    return syms, major


def _extract_file_info(path: str, ext: str) -> Tuple[List[str], set]:
    """(display_symbols, major_names). Symbols are stack-enriched (Angular
    decorators, C# namespace/endpoints); major_names feed the reference graph."""
    text = _read_head(path)
    if not text:
        return [], set()
    res = _ts_symbols(text, ext) if REPOMAP_TREE_SITTER else None
    syms, major = res if res is not None else _regex_symbols(text, ext)

    extra: List[str] = []
    if ext in (".ts", ".tsx"):
        sel = _NG_SELECTOR_RE.search(text)
        for label, rx in _NG_DECORATORS:
            if rx.search(text):
                extra.append(f"@Component({sel.group(1)})"
                             if (label == "@Component" and sel) else label)
                break
    elif ext == ".cs":
        ns = _CS_NAMESPACE_RE.search(text)
        if ns:
            extra.append(f"namespace {ns.group(1)}")
        for verb, route in _CS_HTTP_RE.findall(text)[:6]:
            extra.append(f"[Http{verb}{(' ' + route) if route else ''}]")

    merged = extra + [s for s in syms if s not in extra]
    return merged[:REPOMAP_SYMS_PER_FILE], major


def _pagerank(edges: Dict[str, Dict[str, int]], d: float = 0.85,
              iters: int = 20) -> Dict[str, float]:
    nodes = list(edges)
    n = len(nodes)
    if n == 0:
        return {}
    pr = {x: 1.0 / n for x in nodes}
    outsum = {x: sum(t.values()) for x, t in edges.items()}
    for _ in range(iters):
        nxt = {x: (1.0 - d) / n for x in nodes}
        dangling = 0.0
        for x in nodes:
            s = outsum[x]
            if s == 0:
                dangling += pr[x]
                continue
            share = d * pr[x]
            for tgt, w in edges[x].items():
                if tgt in nxt:
                    nxt[tgt] += share * (w / s)
        if dangling:
            add = d * dangling / n
            for x in nodes:
                nxt[x] += add
        pr = nxt
    return pr


def _centrality(root: str, files: List[Dict[str, Any]]) -> Dict[str, float]:
    """PageRank over the symbol-reference graph: file A -> file B when A mentions
    a major symbol defined in B. Skipped (empty) above REPOMAP_GRAPH_MAX_FILES."""
    if not REPOMAP_GRAPH or len(files) > REPOMAP_GRAPH_MAX_FILES:
        return {}
    defs: Dict[str, set] = {}
    for f in files:
        for mname in f["major"]:
            defs.setdefault(mname, set()).add(f["rel"])
    names = set(defs)
    if not names:
        return {}
    code_files = [f for f in files if f["ext"] in _SYMBOL_PATTERNS]
    rels = {f["rel"] for f in code_files}
    edges: Dict[str, Dict[str, int]] = {f["rel"]: {} for f in code_files}
    for f in code_files:
        text = _read_head(os.path.join(root, f["rel"].replace("/", os.sep)),
                          REPOMAP_GRAPH_READ_BYTES)
        if not text:
            continue
        refs = set(_IDENT_RE.findall(text)) & names
        tgts = edges[f["rel"]]
        for nm in refs:
            for tgt in defs[nm]:
                if tgt != f["rel"] and tgt in rels:
                    tgts[tgt] = tgts.get(tgt, 0) + 1
    return _pagerank(edges)


def _build_index(root: str, sig: Optional[str]) -> Dict[str, Any]:
    """Walk -> classify -> symbols -> centrality -> score. Pure filesystem work."""
    ignore = _REPOMAP_IGNORE_DIRS | set(REPOMAP_EXTRA_IGNORES)
    files: List[Dict[str, Any]] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ignore and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _REPOMAP_INDEX_EXTS and fn not in _REPOMAP_SPECIAL_FILES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            role, weight, detail = _classify(rel)
            syms: List[str] = []
            major: List[str] = []
            if REPOMAP_SYMBOLS and detail and ext in _SYMBOL_PATTERNS:
                s, mj = _extract_file_info(full, ext)
                syms, major = s, sorted(mj)
            files.append({"rel": rel, "ext": ext, "role": role, "w": weight,
                          "detail": detail, "syms": syms, "major": major})
            if len(files) >= REPOMAP_MAX_FILES:
                truncated = True
                break
        if truncated:
            break

    cen = _centrality(root, files)
    maxc = max(cen.values()) if cen else 0.0
    for f in files:
        c = (cen.get(f["rel"], 0.0) / maxc) if maxc > 0 else 0.0
        f["c"] = round(c, 4)
        f["score"] = round(f["w"] * (1.0 + REPOMAP_GRAPH_BOOST * c), 4)
    return {"root": root, "sig": sig, "files": files, "truncated": truncated,
            "n_files": len(files)}


def _emit_tree(kept: List[Dict[str, Any]]) -> str:
    tree: Dict[str, Any] = {}
    for f in sorted(kept, key=lambda x: x["rel"]):
        parts = f["rel"].split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p + "/", {})
        node[parts[-1]] = f["syms"][:REPOMAP_SYMS_PER_FILE]
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
    return "\n".join(lines)


def _render_breadth(files: List[Dict[str, Any]], kept_set: set) -> str:
    """Per-directory counts (with dominant roles) for files NOT shown in detail,
    so the model still sees the whole-repo shape cheaply."""
    dirs: Dict[str, Dict[str, Any]] = {}
    for f in files:
        if f["rel"] in kept_set:
            continue
        d = (f["rel"].rsplit("/", 1)[0] + "/") if "/" in f["rel"] else "./"
        agg = dirs.setdefault(d, {"n": 0, "roles": {}})
        agg["n"] += 1
        agg["roles"][f["role"]] = agg["roles"].get(f["role"], 0) + 1
    if not dirs:
        return ""
    lines: List[str] = []
    budget = 0
    shown = 0
    for d in sorted(dirs):
        agg = dirs[d]
        top = sorted(agg["roles"].items(), key=lambda x: -x[1])[:3]
        line = f"{d} — {agg['n']} files ({', '.join(f'{r}:{n}' for r, n in top)})"
        if budget + len(line) + 1 > REPOMAP_BREADTH_MAX_CHARS and lines:
            break
        budget += len(line) + 1
        lines.append(line)
        shown += 1
    omitted = len(dirs) - shown
    if omitted > 0:
        lines.append(f"... {omitted} more director{'y' if omitted == 1 else 'ies'} omitted.")
    return "\n".join(lines)


def _render_static(index: Dict[str, Any]) -> str:
    """STATIC <repoMap>: role-ranked detail (within budget) + collapsed breadth.
    Deterministic for a given tree state, so the cached prefix stays byte-stable."""
    root = index["root"]
    sig = index["sig"] or "no-git"
    files = index["files"]
    total = index["n_files"]

    ranked = sorted((f for f in files if f["detail"]),
                    key=lambda f: (-f["score"], f["rel"]))
    kept: List[Dict[str, Any]] = []
    budget = 0
    for f in ranked:
        cost = len(f["rel"]) + sum(len(s) + 2 for s in f["syms"][:REPOMAP_SYMS_PER_FILE]) + 8
        if budget + cost > REPOMAP_MAX_CHARS and kept:
            break
        budget += cost
        kept.append(f)
    kept_set = {f["rel"] for f in kept}

    parts = [
        f'<repoMap root="{root}">',
        (f"{len(kept)} of {total} files{'+' if index['truncated'] else ''} shown in "
         f"detail, ranked by role + reference centrality ({sig}). Paths are "
         f"relative to the root; top-level symbols follow each file. Use this to "
         f"locate code directly instead of searching."),
        _emit_tree(kept),
    ]
    if REPOMAP_BREADTH:
        breadth = _render_breadth(files, kept_set)
        if breadth:
            parts.append("Directory overview (files not detailed above):")
            parts.append(breadth)
    parts.append("</repoMap>")
    return "\n".join(parts)


def _retrieve(question: str, indexes: List[Dict[str, Any]], k: int) -> List[Tuple]:
    """Deterministic retrieval: score every file against the question by sub-token
    overlap with its symbols (3x) and path (2x), nudged by centrality and role.
    Returns up to k (score, file, root) tuples, best first."""
    q = _subtokens(question) - _STOPWORDS
    if not q:
        return []
    scored: List[Tuple] = []
    for index in indexes:
        root = index["root"]
        for f in index["files"]:
            stok = set()
            for s in f["syms"]:
                stok |= _subtokens(s)
            for mname in f["major"]:
                stok |= _subtokens(mname)
            sym_hits = len(q & stok)
            path_hits = len(q & _subtokens(f["rel"]))
            if sym_hits == 0 and path_hits == 0:
                continue
            sc = 3.0 * sym_hits + 2.0 * path_hits + 0.5 * f.get("c", 0.0) + 0.05 * f["w"]
            scored.append((sc, f, root))
    scored.sort(key=lambda x: (-x[0], x[1]["rel"]))
    return scored[:k]


def _render_hint(hits: List[Tuple]) -> str:
    lines = ["<relevantFiles>",
             "Files most relevant to this request (deterministic retrieval over "
             "the repo index — confirm before relying):"]
    for _sc, f, root in hits:
        label = os.path.basename(root.rstrip("/\\")) or root
        syms = ", ".join(f["syms"][:6])
        lines.append(f"- {label}/{f['rel']}" + (f" — {syms}" if syms else ""))
    lines.append("</relevantFiles>")
    return "\n".join(lines)


def _repomap_disk_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(REPOMAP_DIR, f"{h}.json")


def _load_index_disk(key: str) -> Optional[Dict[str, Any]]:
    path = _repomap_disk_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("key") == key and "index" in d and "static_text" in d:
            return {"sig": d.get("sig"), "built_at": float(d.get("built_at", 0.0)),
                    "index": d["index"], "static_text": d["static_text"]}
    except Exception:
        pass
    return None


def _save_index_disk(key: str, ent: Dict[str, Any]):
    try:
        os.makedirs(REPOMAP_DIR, exist_ok=True)
        with open(_repomap_disk_path(key), "w", encoding="utf-8") as f:
            json.dump({"key": key, "sig": ent["sig"], "built_at": ent["built_at"],
                       "index": ent["index"], "static_text": ent["static_text"]},
                      f, ensure_ascii=False)
    except Exception as e:
        print(f"[qwen-server] repo index disk save failed: {e!r}")


def _get_index(root: str) -> Optional[Dict[str, Any]]:
    """Cached per-root index entry {sig, built_at, index, static_text}. Fresh
    within TTL -> served from memory (no walk). Else recheck git HEAD: reuse if
    unchanged, rebuild otherwise. Falls back to disk across restarts. None if the
    path isn't a real dir on this host."""
    norm = os.path.normpath(root)
    if not os.path.isdir(norm):
        return None
    key = os.path.normcase(norm)
    now = time.time()

    ent = _REPOMAP_CACHE.get(key)
    if ent and now - ent["built_at"] < REPOMAP_TTL:
        return ent
    if ent is None:
        ent = _load_index_disk(key)
        if ent:
            _REPOMAP_CACHE[key] = ent

    sig = _repo_signature(norm)
    if ent and sig is not None and ent.get("sig") == sig:
        ent["built_at"] = now
        return ent
    if ent and sig is None and now - ent["built_at"] < REPOMAP_TTL:
        return ent

    t0 = time.time()
    index = _build_index(norm, sig)
    static_text = _render_static(index)
    ent = {"sig": sig, "built_at": now, "index": index, "static_text": static_text}
    _REPOMAP_CACHE[key] = ent
    _save_index_disk(key, ent)
    print(f"[qwen-server] repo index BUILT for {norm}: {index['n_files']} files, "
          f"static {len(static_text)}c, {sig or 'no-git'}, {time.time() - t0:.1f}s")
    return ent


def _inject_repo_map(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inject (1) the STATIC map into the <workspace_info> message (cached prefix)
    and (2) the DYNAMIC retrieval hint before <userRequest> (volatile tail), for
    EVERY workspace root. Pure Python; runs in the request handler before
    tokenization. No-op for requests with no workspace block."""
    if not REPOMAP_ENABLED:
        return messages
    roots = _workspace_roots(messages)
    if not roots:
        return messages

    indexes: List[Dict[str, Any]] = []
    statics: List[str] = []
    for r in roots:
        try:
            ent = _get_index(r)
        except Exception as e:
            print(f"[qwen-server] repo index failed for {r!r}: {e!r}")
            ent = None
        if ent:
            indexes.append(ent["index"])
            statics.append(ent["static_text"])
    if not statics:
        return messages

    # (1) STATIC -> append to the <workspace_info> message (before <context>).
    static_block = "\n".join(statics)
    out: List[Dict[str, Any]] = []
    done = False
    for m in messages:
        c = m.get("content")
        if not done and isinstance(c, str) and "following folders" in c:
            m = {**m, "content": c + "\n" + static_block}
            done = True
        out.append(m)
    messages = out

    # (2) DYNAMIC -> insert before the last <userRequest> (after <context>).
    n_hint = 0
    if REPOMAP_HINT and indexes:
        q = _extract_question(messages)
        hits = _retrieve(q, indexes, REPOMAP_HINT_FILES) if q else []
        if hits:
            idx = -1
            for i, m in enumerate(messages):
                c = m.get("content")
                if isinstance(c, str) and "<userRequest>" in c:
                    idx = i
            if idx >= 0:
                hint = _render_hint(hits)
                c = messages[idx]["content"]
                messages[idx] = {**messages[idx],
                                 "content": c.replace("<userRequest>",
                                                      hint + "\n<userRequest>", 1)}
                n_hint = len(hits)

    print(f"[qwen-server] repo map: {len(statics)} root(s), static {len(static_block)}c"
          + (f", hint {n_hint} file(s)" if n_hint else ", no hint"))
    return messages


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