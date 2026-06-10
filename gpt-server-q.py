#!/usr/bin/env python3
"""
qwen-server-mlx.py — OpenAI-compatible server for Qwen3.6-35B-A3B (MLX 4-bit)
on Apple Silicon, drop-in replacement for qwen-server-gguf.py as a GitHub
Copilot Custom Endpoint (same URL, same tool calling, same <think> handling).

WHY MLX (vs the llama-cpp-python version)
  The GGUF server's prefix reuse was implicit (one hidden slot inside
  llama-cpp-python) and was observed re-prefilling the full ~30k Copilot
  prompt every turn despite >95% token-level prefix match. Here the prompt
  cache is EXPLICIT and owned by this script:

  1. SLOT CACHE. QWEN_CACHE_SLOTS (default 2) independent prompt caches.
     Each request is routed to the slot with the longest common token prefix,
     so a Copilot side request (title gen, summarization) lands in its own
     slot instead of evicting the agent loop's 30k-token prefix.

  2. EXACT TRIM + DELTA PREFILL. The chosen slot's cache is trimmed to the
     common prefix (mlx_lm trim_prompt_cache), then ONLY the new suffix is
     prefilled. After every request the slot's token ledger is synced from
     the cache's actual offset — ground truth, not bookkeeping hope — so a
     crash or partial generation can never silently corrupt reuse.

  3. VISIBILITY. Every request logs: slot id, prompt tokens, reused tokens,
     suffix actually prefilled, TTFT, prefill t/s (from MLX itself), and
     decode t/s. If reuse ever drops, you'll see exactly when and how much.

  KV cache is fp16 (Qwen3 MoE KV is small; ~30k tokens is a few GB per slot,
  fine on 64GB next to ~18GB of 4-bit weights). Sampling matches the GGUF
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
  export QWEN_MLX="mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"   # or a local dir
  python qwen-server-mlx.py                                    # http://127.0.0.1:8000

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import (
    make_prompt_cache,
    trim_prompt_cache,
    can_trim_prompt_cache,
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get("QWEN_MLX", "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit")
MODEL_ID = os.environ.get("QWEN_ID", "qwen3.6-35b-a3b")
PORT = int(os.environ.get("QWEN_PORT", "8000"))
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "4096"))

# Prefill chunk size MLX pushes through Metal per step. 2048 mirrors the GGUF
# server's n_ubatch; try 4096 on M4 Max (cheap A/B, watch the prefill t/s log).
PREFILL_STEP = int(os.environ.get("QWEN_PREFILL_STEP", "2048"))

# Prompt-cache slots. 3 = agent loop + two side-request streams (Copilot can
# fire title gen AND summaries between agent turns) without mutual eviction.
# Each slot holds fp16 KV for its full prompt (~a few GB at 30k tokens).
CACHE_SLOTS = int(os.environ.get("QWEN_CACHE_SLOTS", "3"))
# A request "matches" a slot only if it shares at least this many prefix
# tokens; otherwise it's treated as a new conversation (fresh/LRU slot).
# NOTE: kv-reuse=0 in the log can therefore mean "matched < MIN_PREFIX",
# not literally zero — the divergence log (below) shows the true number.
MIN_PREFIX = int(os.environ.get("QWEN_MIN_PREFIX", "1024"))

# Cache diagnostics: when a request diverges from its best-matching slot,
# print WHERE it diverged and the differing text on both sides. This is the
# tool for hunting prompt churn (timestamps, shuffled tools, editor state).
CACHE_DEBUG = os.environ.get("QWEN_CACHE_DEBUG", "1") == "1"

# Copilot is not guaranteed to send the tools array in a stable order, and
# tools render near the TOP of the Qwen prompt — a shuffled tool list breaks
# ALL prefix reuse by itself. Sorting by function name makes the rendered
# prompt deterministic; the model just sees a consistent ordering.
SORT_TOOLS = os.environ.get("QWEN_SORT_TOOLS", "1") == "1"

# Optional: ask MLX to wire its GPU buffers (resists page eviction between
# turns). Set to e.g. 56 (GB). Requires the sysctl above to be raised first.
WIRED_GB = float(os.environ.get("QWEN_WIRED_GB", "0"))

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


def _load():
    _set_wired_limit()
    print(f"[qwen-server] loading MLX model {MODEL_PATH} ...")
    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer


model, tok = GPU.submit(_load).result()
print(f"[qwen-server] ready. model id: {MODEL_ID} "
      f"(slots={CACHE_SLOTS}, prefill_step={PREFILL_STEP})")

app = FastAPI()


# --------------------------------------------------------------------------- #
# Prompt-cache slots (explicit, observable prefix reuse)
# --------------------------------------------------------------------------- #
class _Slot:
    __slots__ = ("idx", "cache", "tokens", "last_used")

    def __init__(self, idx: int):
        self.idx = idx
        self.cache = make_prompt_cache(model)
        self.tokens: List[int] = []   # ledger of tokens the cache contains
        self.last_used = 0.0

    def reset(self):
        self.cache = make_prompt_cache(model)
        self.tokens = []


_SLOTS: List[_Slot] = []


def _common_prefix(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _log_divergence(slot: "_Slot", tokens: List[int], k: int):
    """Print where the new prompt stops matching the best slot, with the text
    on both sides of the split. One glance identifies the churn source:
    a timestamp, shuffled tool defs, editor state, etc."""
    if k >= min(len(tokens), len(slot.tokens)):
        return  # clean extension, nothing diverged
    try:
        lo = max(0, k - 15)
        cached = tok.decode(slot.tokens[lo:k + 40])
        new = tok.decode(tokens[lo:k + 40])
        pct = (100 * k // len(tokens)) if tokens else 0
        print(f"[qwen-server] slot={slot.idx} best match {k}t/{len(tokens)}t "
              f"({pct}%) — diverges at token {k}:\n"
              f"  cached: {cached!r}\n"
              f"  new:    {new!r}")
    except Exception:
        pass


def _acquire_slot(tokens: List[int]) -> Tuple[_Slot, int]:
    """Pick the slot with the longest common prefix; spawn or LRU-recycle a
    slot if nothing matches meaningfully. Returns (slot, usable_prefix_len).
    Worker thread only."""
    best, best_k = None, -1
    for s in _SLOTS:
        k = _common_prefix(s.tokens, tokens)
        if k > best_k:
            best, best_k = s, k
    if CACHE_DEBUG and best is not None and best.tokens:
        _log_divergence(best, tokens, best_k)
    if best is not None and best_k >= MIN_PREFIX:
        return best, best_k
    if len(_SLOTS) < CACHE_SLOTS:
        s = _Slot(len(_SLOTS))
        _SLOTS.append(s)
        return s, 0
    # All slots busy and nothing matched well: recycle the LRU slot, but keep
    # whatever prefix it DOES share with this prompt instead of starting cold.
    s = min(_SLOTS, key=lambda x: x.last_used)
    k = _common_prefix(s.tokens, tokens)
    if k > 0:
        return s, k
    s.reset()
    return s, 0


def _prepare_slot(slot: _Slot, tokens: List[int], k: int) -> Tuple[List[int], int]:
    """Trim the slot's cache down to the common prefix and return the suffix
    that still needs prefilling. Always leaves >=1 token to evaluate (MLX
    needs at least one forward pass to produce logits)."""
    if k >= len(tokens):
        k = len(tokens) - 1
    n_trim = len(slot.tokens) - k
    if n_trim > 0:
        if can_trim_prompt_cache(slot.cache):
            trim_prompt_cache(slot.cache, n_trim)
            slot.tokens = slot.tokens[:k]
        else:
            slot.reset()
            k = 0
    return tokens[k:], k


def _sync_slot(slot: _Slot, prompt_tokens: List[int], gen_ids: List[int]):
    """Sync the slot ledger from the cache's ACTUAL offset after generation.
    stream_generate's final sampled token is never evaluated into the cache
    (and the last loop token can be double-reported), so instead of trusting
    arithmetic we read cache[0].offset as ground truth and take that prefix
    of (prompt + generated). Any inconsistency -> reset rather than risk a
    corrupted reuse next turn."""
    try:
        n_cached = int(slot.cache[0].offset)
        full = prompt_tokens + gen_ids
        if 0 <= n_cached <= len(full):
            slot.tokens = full[:n_cached]
            if CACHE_DEBUG:
                print(f"[qwen-server] slot={slot.idx} cache holds {n_cached}t")
        else:
            print(f"[qwen-server] WARNING slot={slot.idx}: cache offset "
                  f"{n_cached} > tracked {len(full)} — resetting, reuse lost")
            slot.reset()
    except Exception as e:
        print(f"[qwen-server] WARNING slot={slot.idx}: ledger sync failed "
              f"({e!r}) — resetting, reuse lost")
        slot.reset()
    slot.last_used = time.time()


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
            return render(**attempt)
        except TypeError:
            continue
    return render()


def _starts_in_think(prompt: str) -> bool:
    """True if the template already opened a <think> block at the end of the
    prompt, so generation starts mid-reasoning with no opening tag emitted."""
    last_open = prompt.rfind("<think>")
    if last_open == -1:
        return False
    return prompt.find("</think>", last_open) == -1


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
    slot, k = _acquire_slot(tokens)
    suffix, k = _prepare_slot(slot, tokens, k)
    n_prompt = len(tokens)
    pct = (100 * k // n_prompt) if n_prompt else 0
    print(f"[qwen-server] slot={slot.idx} prompt={n_prompt}t "
          f"kv-reuse={k}t ({pct}%) prefill={len(suffix)}t")

    sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
    processors = make_logits_processors(
        repetition_penalty=repeat if repeat and repeat != 1.0 else None,
        repetition_context_size=256,
    ) or []
    if presence:
        processors.append(_presence_processor(presence))

    t0 = time.perf_counter()
    t_first = None
    n_out = 0
    gen_ids: List[int] = []
    try:
        for resp in stream_generate(
            model, tok, prompt=suffix,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
            prompt_cache=slot.cache,
            prefill_step_size=PREFILL_STEP,
        ):
            gen_ids.append(resp.token)
            piece = resp.text
            if piece:
                if t_first is None:
                    t_first = time.perf_counter()
                    dt = t_first - t0
                    tps = getattr(resp, "prompt_tps", 0) or 0
                    print(f"[qwen-server] TTFT={dt:.2f}s"
                          + (f" (prefill ~{tps:.0f} t/s)" if tps else ""))
                n_out += 1
                yield piece
    finally:
        # Always resync the ledger from the cache's real state, even on error.
        _sync_slot(slot, tokens, gen_ids)

    t_end = time.perf_counter()
    if t_first is not None and n_out > 1 and t_end > t_first:
        print(f"[qwen-server] decode: {n_out} chunks in {t_end - t_first:.2f}s "
              f"(~{n_out / (t_end - t_first):.1f} t/s)")


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

        reasoning, content, tool_calls = await asyncio.wrap_future(GPU.submit(generate_all))
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
                q.put(("err", repr(e)))
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
        while True:
            kind, val = q.get()
            if kind == "end":
                break
            if kind == "err":
                print(f"[qwen-server] generation error: {val}")
                errored = True
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