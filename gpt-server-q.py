#!/usr/bin/env python3
"""
qwen-server-gguf.py — OpenAI-compatible server for Qwen3.6-35B-A3B (GGUF) on
Apple Silicon via llama-cpp-python, built so GitHub Copilot's Custom Endpoint
treats it like a native model: tool calling, agent/plan mode, clean formatting
(reasoning never leaks into the answer).

PERF NOTES (this revision — tuned for Copilot's ~30k-token agent prompts)
  The dominant cost is PREFILL of the huge Copilot prompt, not decode. Three
  levers, in order of impact:

  1. KV-prefix reuse (free, already structural). Everything runs on ONE Llama
     instance on ONE worker thread, so llama.cpp automatically diffs each new
     prompt against the tokens already in the KV cache and only prefills the
     new tail. Agent loops replay the same prefix + new turns, so after the
     first (slow) request, each turn should prefill only the delta.
     -> run_text() now LOGS "kv-reuse" per request. If you see reuse ~0% on
        agent turns, Copilot is injecting changing content (timestamps, open
        editor context) near the TOP of the prompt and killing the cache —
        that's the thing to hunt, not sampling knobs.

  2. Prefill throughput: n_batch AND n_ubatch raised to 2048 (n_ubatch is the
     one that actually gates Metal prefill chunking; llama-cpp-python defaults
     it to 512 regardless of n_batch). KV cache defaults to q8_0 now
     (flash_attn is on, which q8 KV requires) — ~half the KV memory at 64k ctx,
     near-lossless, less memory pressure on a 64GB box.

  3. Decode: prompt-lookup speculative decoding now defaults ON (coding/agent
     output echoes the prompt heavily — file contents, identifiers — so the
     n-gram draft hits a lot). QWEN_PROMPT_LOOKUP=0 to disable if it ever
     regresses. QWEN_THINKING=0 disables <think> globally for max agent speed
     (plan-mode quality tradeoff; per-request reasoning_effort still wins).

  Optional: QWEN_RAM_CACHE=1 enables LlamaRAMCache so two *interleaved*
  conversations (e.g. Copilot firing a side request mid-agent-loop) don't
  evict each other's prefix. Off by default: it snapshots full KV state per
  entry, which is GB-scale at long context — watch memory if you turn it on.

WHAT CHANGED vs the first draft (and WHY agent mode was failing)
  Qwen3.6-35B-A3B was trained on the Qwen3-CODER tool-call format, which is
  nested XML, NOT Hermes JSON:

      <tool_call>
      <function=read_file>
      <parameter=path>
      src/main.py
      </parameter>
      </function>
      </tool_call>

  The old parser ran json.loads() on that, threw, swallowed the error, and
  emitted a tool call with name="" + raw XML as arguments. Copilot discards
  empty-name tool calls, so the model would narrate its plan ("I'll explore the
  codebase...") and then stop with no tool execution. The parser below reads the
  real XML format (with a JSON fallback in case a turn emits Hermes style),
  does schema-aware type coercion of parameter values, and never emits a
  nameless tool call.

  Also fixed / added:
   - finish_reason="tool_calls" is now sent in its OWN terminal chunk (some
     strict clients choke on a delta + finish_reason in the same SSE chunk).
   - Unclosed tool calls (cut off by max_tokens / EOS) are still finalized
     instead of being silently dropped.
   - presence_penalty default (Qwen3.6's recommended anti-loop knob).
   - preserve_thinking chat_template_kwarg on by default (helps agent turns).

WHY THIS FILE (vs the MLX one): MLX can't read a GGUF k-quant, so to use the
.gguf you already have from Python you need llama-cpp-python (prebuilt Metal
wheels exist for Apple Silicon; it's llama.cpp under the hood).

RUN
  python3 -m venv ~/llama-env && source ~/llama-env/bin/activate
  pip install -U llama-cpp-python transformers fastapi "uvicorn[standard]"
  export QWEN_GGUF="$HOME/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
  export QWEN_TOKENIZER="Qwen/Qwen3.6-35B-A3B"   # tokenizer/template only
  python qwen-server-gguf.py                     # http://127.0.0.1:8000
  (CPU-only wheel? force Metal:
     CMAKE_ARGS="-DGGML_METAL=on" pip install -U --no-binary llama-cpp-python llama-cpp-python)

VS CODE (chatLanguageModels.json): point `url` at
  http://localhost:8000/v1/chat/completions , set "toolCalling": true, "thinking": true
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

import numpy as np
from llama_cpp import Llama
from transformers import AutoTokenizer

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
GGUF_PATH = os.environ.get("QWEN_GGUF", os.path.expanduser("~/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"))
TOKENIZER = os.environ.get("QWEN_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
MODEL_ID = os.environ.get("QWEN_ID", "qwen3.6-35b-a3b")
PORT = int(os.environ.get("QWEN_PORT", "8000"))
N_CTX = int(os.environ.get("QWEN_CTX", "65536"))         # model is 262K-native; raise if RAM allows
N_GPU_LAYERS = int(os.environ.get("QWEN_NGL", "-1"))     # -1 = offload all to Metal

# Prefill batching. n_ubatch is the micro-batch llama.cpp actually pushes
# through Metal per step; llama-cpp-python defaults it to 512 even when
# n_batch is larger, so set BOTH. 2048 is a good M4 Max starting point.
N_BATCH = int(os.environ.get("QWEN_N_BATCH", "2048"))
N_UBATCH = int(os.environ.get("QWEN_N_UBATCH", str(N_BATCH)))
N_THREADS = os.environ.get("QWEN_N_THREADS")             # set to P-core count (M4 Max: ~12); None = auto
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "4096"))

# KV-cache quantization. 0=f16, 8=q8_0, 2=q4_0. Default is now q8_0: nearly
# free quality-wise, roughly halves KV memory at 64k ctx (requires flash_attn,
# which is on). Set QWEN_KV_TYPE_K=0 QWEN_KV_TYPE_V=0 to go back to f16.
KV_TYPE_K = int(os.environ.get("QWEN_KV_TYPE_K", "8"))
KV_TYPE_V = int(os.environ.get("QWEN_KV_TYPE_V", "8"))

# Prompt-lookup speculative decoding — ON by default. Coding/agent output
# echoes the input (file contents, identifiers) heavily, so the n-gram draft
# model hits often and decode speeds up for free. QWEN_PROMPT_LOOKUP=0 to off.
USE_PROMPT_LOOKUP = os.environ.get("QWEN_PROMPT_LOOKUP", "1") == "1"
PROMPT_LOOKUP_TOKENS = int(os.environ.get("QWEN_PROMPT_LOOKUP_TOKENS", "10"))

# Global thinking kill-switch. QWEN_THINKING=0 renders the template with
# enable_thinking=False on every request — big latency win in agent loops at
# some planning-quality cost. Per-request enable_thinking/reasoning_effort
# from the client still take precedence.
THINKING_DEFAULT = os.environ.get("QWEN_THINKING", "1") == "1"

# Optional cross-conversation prefix cache (see header). Off by default.
USE_RAM_CACHE = os.environ.get("QWEN_RAM_CACHE", "0") == "1"
RAM_CACHE_GB = float(os.environ.get("QWEN_RAM_CACHE_GB", "16"))

# Per-request chat logging for visibility: writes one JSON per request into
# ./chats (relative to wherever you launch the server), containing the raw
# OpenAI messages, the EXACT rendered prompt string fed to llama.cpp, and the
# full response (reasoning / content / tool calls). Toggle: QWEN_CHAT_LOG=0.
CHAT_LOG = os.environ.get("QWEN_CHAT_LOG", "1") == "1"
CHAT_LOG_DIR = os.environ.get("QWEN_CHAT_LOG_DIR", os.path.join(os.getcwd(), "chats"))

# Qwen3.6 thinking-mode sampling. Coding-leaning defaults; presence_penalty is
# the recommended anti-loop knob (Qwen3.6 suggests up to ~1.5 in thinking mode).
DEF_TEMP = float(os.environ.get("QWEN_TEMP", "0.7"))
DEF_TOP_P = float(os.environ.get("QWEN_TOP_P", "0.8"))
DEF_TOP_K = int(os.environ.get("QWEN_TOP_K", "20"))
DEF_MIN_P = float(os.environ.get("QWEN_MIN_P", "0.0"))
DEF_REPEAT = float(os.environ.get("QWEN_REPEAT_PENALTY", "1.05"))
DEF_PRESENCE = float(os.environ.get("QWEN_PRESENCE_PENALTY", "1.5"))
DEF_FREQUENCY = float(os.environ.get("QWEN_FREQUENCY_PENALTY", "0.0"))

# llama-cpp-python's Llama object is not safe for concurrent calls, and a stream
# generator must be drained on the thread that created it. Pin all model work to
# one worker; this also serializes overlapping requests AND lets llama.cpp reuse
# the shared KV prefix across agent turns (free prefix caching). NOTE: that
# automatic reuse is "previous request only" — if two conversations interleave,
# they evict each other. QWEN_RAM_CACHE=1 papers over that at a memory cost.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")


def _load():
    print(f"[qwen-server] loading GGUF {GGUF_PATH} (n_ctx={N_CTX}, ngl={N_GPU_LAYERS}, "
          f"n_batch={N_BATCH}, n_ubatch={N_UBATCH}, kv={KV_TYPE_K}/{KV_TYPE_V}) ...")
    kw: Dict[str, Any] = dict(
        model_path=GGUF_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH, n_ubatch=N_UBATCH, flash_attn=True, verbose=False,
    )
    if N_THREADS:
        kw["n_threads"] = int(N_THREADS)
    if KV_TYPE_K:
        kw["type_k"] = KV_TYPE_K
    if KV_TYPE_V:
        kw["type_v"] = KV_TYPE_V
    if USE_PROMPT_LOOKUP:
        try:
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
            kw["draft_model"] = LlamaPromptLookupDecoding(num_pred_tokens=PROMPT_LOOKUP_TOKENS)
            print(f"[qwen-server] prompt-lookup decoding on (n={PROMPT_LOOKUP_TOKENS})")
        except Exception as e:
            print(f"[qwen-server] prompt-lookup unavailable, continuing without: {e!r}")
    model = Llama(**kw)
    if USE_RAM_CACHE:
        try:
            from llama_cpp import LlamaRAMCache
            model.set_cache(LlamaRAMCache(capacity_bytes=int(RAM_CACHE_GB * (1 << 30))))
            print(f"[qwen-server] RAM prefix cache on ({RAM_CACHE_GB:g} GB) — watch memory")
        except Exception as e:
            print(f"[qwen-server] RAM cache unavailable, continuing without: {e!r}")
    return model


llm = GPU.submit(_load).result()
tok = AutoTokenizer.from_pretrained(TOKENIZER)
print(f"[qwen-server] ready. model id: {MODEL_ID}")

app = FastAPI()


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
    the well-known 'Can only get item pairs from a mapping' template crash).
    Preserve reasoning_content on assistant turns so preserve_thinking has
    something to keep if the client replays it (harmless otherwise)."""
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

    enable_thinking = body.get("enable_thinking")
    if enable_thinking is None:
        effort = str(body.get("reasoning_effort", "")).lower()
        if effort == "none":
            enable_thinking = False
        elif effort in ("low", "medium", "high"):
            enable_thinking = True
        else:
            enable_thinking = THINKING_DEFAULT

    # preserve_thinking helps Qwen3.6 in multi-turn agent loops; let the client
    # override via chat_template_kwargs.
    extra = {"preserve_thinking": True}
    extra.update(dict(body.get("chat_template_kwargs") or {}))

    def render(**kw) -> str:
        return tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw
        )

    # Degrade gracefully if a given template doesn't accept a kwarg.
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
# Qwen3-Coder XML tool-call parsing
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
    """Coerce a raw XML parameter string into the type the tool schema expects.
    The model emits everything as text; Copilot validates against the schema, so
    a string '3' for an integer param (or 'true' for a bool) must be converted."""
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
        # Unknown/absent schema: best-effort JSON, else keep as text.
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
    """Parse the inside of a <tool_call>...</tool_call> block.
    Tries Qwen3-Coder XML first, falls back to Hermes JSON."""
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

    # Fallback: Hermes JSON ({"name": ..., "arguments": {...}})
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
    """Qwen emits optional <think>...</think> first, then content that may
    contain <tool_call>...</tool_call> blocks (Coder XML inside). feed(delta)
    returns ("reasoning"|"content", str) events to stream now; finish() flushes
    tool calls."""

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
# Inference: stream text deltas out of llama.cpp (+ perf instrumentation)
# --------------------------------------------------------------------------- #
def _prefix_reuse_stats(prompt: str) -> Tuple[int, int]:
    """(n_prompt_tokens, n_reused_from_kv). Mirrors the comparison llama.cpp
    does internally so we can SEE whether Copilot's prompt is cache-friendly.
    Best-effort: any failure just disables the metric."""
    try:
        toks = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        n_prompt = len(toks)
        prev = np.asarray(llm._input_ids[: llm.n_tokens]) if llm.n_tokens else np.empty(0, dtype=np.int64)
        n = min(len(prev), n_prompt)
        if n == 0:
            return n_prompt, 0
        cur = np.asarray(toks[:n])
        mismatch = np.nonzero(prev[:n] != cur)[0]
        reused = int(mismatch[0]) if mismatch.size else n
        return n_prompt, reused
    except Exception:
        return -1, 0


def run_text(prompt: str, body: Dict[str, Any]):
    """Yield generated text deltas (worker thread only)."""
    def pick(key, default, cast):
        v = body.get(key)
        return default if v is None else cast(v)

    temp = pick("temperature", DEF_TEMP, float)
    top_p = pick("top_p", DEF_TOP_P, float)
    top_k = pick("top_k", DEF_TOP_K, int)
    min_p = pick("min_p", DEF_MIN_P, float)
    repeat = pick("repetition_penalty", DEF_REPEAT, float)
    presence = pick("presence_penalty", DEF_PRESENCE, float)
    frequency = pick("frequency_penalty", DEF_FREQUENCY, float)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or DEFAULT_MAX_TOKENS)

    n_prompt, reused = _prefix_reuse_stats(prompt)
    to_prefill = (n_prompt - reused) if n_prompt >= 0 else -1
    if n_prompt >= 0:
        pct = (100 * reused // n_prompt) if n_prompt else 0
        print(f"[qwen-server] prompt={n_prompt}t kv-reuse={reused}t ({pct}%) prefill~{to_prefill}t")

    t0 = time.perf_counter()
    t_first = None
    n_out = 0
    for out in llm.create_completion(
        prompt, max_tokens=max_tokens, temperature=temp, top_p=top_p,
        top_k=top_k, min_p=min_p, repeat_penalty=repeat,
        presence_penalty=presence, frequency_penalty=frequency,
        stop=["<|im_end|>"], stream=True,
    ):
        piece = out["choices"][0].get("text", "")
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
                dt = t_first - t0
                rate = (to_prefill / dt) if (to_prefill > 0 and dt > 0) else 0
                print(f"[qwen-server] TTFT={dt:.2f}s"
                      + (f" (~{rate:.0f} t/s prefill)" if rate else ""))
            n_out += 1
            yield piece
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
    """Dump everything the model saw and everything it produced for one
    request. 'rendered_prompt' is the literal string handed to llama.cpp
    (post chat-template), i.e. exactly what the model sees. Best-effort:
    logging failures never break a request."""
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
            for delta in run_text(prompt, body):
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
    # deltas to a queue; the SSE generator only drains the queue and parses
    # (pure Python, thread-safe). No llama.cpp call leaves the worker.
    def sse():
        parser = QwenStreamParser(initial_mode=init_mode, tools=tools)
        q: "queue.Queue" = queue.Queue(maxsize=256)
        acc_reasoning, acc_content = [], []   # accumulated for the chat log

        def produce():
            try:
                for delta in run_text(prompt, body):
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