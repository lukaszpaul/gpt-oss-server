#!/usr/bin/env python3
"""
qwen-server-gguf.py — OpenAI-compatible server for Qwen3.6-35B-A3B (GGUF) on
Apple Silicon via llama-cpp-python, built so GitHub Copilot's Custom Endpoint
treats it like a native model: tool calling, agent/plan mode, clean formatting
(reasoning never leaks into the answer).

WHY THIS FILE EXISTS (vs the MLX one)
  MLX cannot read a GGUF k-quant (Q4_K_M). The only way to use the .gguf you
  already have, from Python, is llama-cpp-python — which IS llama.cpp compiled
  into a pip-installable extension. There is no separate llama-server binary and
  nothing to build from source (prebuilt Metal wheels exist for Apple Silicon),
  but it is llama.cpp under the hood. If that's acceptable, this is your path; if
  the llama.cpp codebase itself is off-limits, download MLX weights and use
  qwen-server.py instead.

  Design: we render the prompt with Qwen's official chat template (via the HF
  tokenizer — a small download, NOT the weights), run raw text generation through
  the GGUF, and split the output stream into reasoning_content / content /
  tool_calls with the same parser used by the MLX version.

RUN
  python3 -m venv ~/llama-env && source ~/llama-env/bin/activate
  # Metal wheel for Apple Silicon (prebuilt; no compiler needed in most setups):
  pip install -U llama-cpp-python transformers fastapi "uvicorn[standard]"
  export QWEN_GGUF="$HOME/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"   # your .gguf file
  export QWEN_TOKENIZER="Qwen/Qwen3.6-35B-A3B"                  # tokenizer/template only
  python qwen-server-gguf.py                                    # http://localhost:8000

  (If pip pulls a CPU-only wheel, force a Metal build:
     CMAKE_ARGS="-DGGML_METAL=on" pip install -U --no-binary llama-cpp-python llama-cpp-python)

VS CODE (chatLanguageModels.json) — point the model's `url` at:
  http://localhost:8000/v1/chat/completions
  set "id" to QWEN_ID, "toolCalling": true, "thinking": true
"""

import os
import json
import time
import uuid
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

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
N_CTX = int(os.environ.get("QWEN_CTX", "32768"))
N_GPU_LAYERS = int(os.environ.get("QWEN_NGL", "-1"))      # -1 = offload all to Metal
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "4096"))

# Qwen3.6 thinking-mode sampling recipe (greedy/low-temp can loop; these match
# Qwen's recommended thinking defaults). Clients may override per request.
DEF_TEMP = float(os.environ.get("QWEN_TEMP", "0.6"))
DEF_TOP_P = float(os.environ.get("QWEN_TOP_P", "0.95"))
DEF_TOP_K = int(os.environ.get("QWEN_TOP_K", "20"))
DEF_MIN_P = float(os.environ.get("QWEN_MIN_P", "0.0"))
DEF_REPEAT = float(os.environ.get("QWEN_REPEAT_PENALTY", "1.0"))   # 1.0 = off

# llama-cpp-python's Llama object is not safe for concurrent calls, and a stream
# generator must be drained on the thread that created it. Pin all model work to
# one worker; this also serializes overlapping requests.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")


def _load():
    print(f"[qwen-server] loading GGUF {GGUF_PATH} (n_ctx={N_CTX}, ngl={N_GPU_LAYERS}) ...")
    return Llama(model_path=GGUF_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS,
                 flash_attn=True, verbose=False)


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
    """Flatten list-style content, and convert OpenAI tool_call JSON-string
    arguments into dicts (Qwen's template iterates the object)."""
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
        enable_thinking = str(body.get("reasoning_effort", "")).lower() != "none"
    extra = dict(body.get("chat_template_kwargs") or {})

    def render(**kw) -> str:
        return tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw
        )

    try:
        return render(enable_thinking=enable_thinking, **extra)
    except TypeError:
        return render(**extra)


def _starts_in_think(prompt: str) -> bool:
    """True if the template already opened a <think> block at the end of the
    prompt (some Qwen templates do this in thinking mode), so generation starts
    mid-reasoning with no opening tag in the model's output."""
    last_open = prompt.rfind("<think>")
    if last_open == -1:
        return False
    return prompt.find("</think>", last_open) == -1


# --------------------------------------------------------------------------- #
# Streaming parser: Qwen text stream -> reasoning / content / tool_calls
# --------------------------------------------------------------------------- #
class QwenStreamParser:
    """Qwen emits optional <think>...</think> first, then content that may
    contain <tool_call>\n{json}\n</tool_call> blocks. feed(delta) returns
    ("reasoning"|"content", str) events to stream now; finish() flushes tool calls."""

    _MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
    _HOLD = max(len(m) for m in _MARKERS) - 1

    def __init__(self, initial_mode: str = "content"):
        self.buf = ""
        self.mode = initial_mode          # "content" | "reasoning"
        self.in_tool = False
        self._tool_buf = ""
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
        raw = self._tool_buf.strip()
        self._tool_buf = ""
        name, args = "", raw
        try:
            obj = json.loads(raw)
            name = obj.get("name", "")
            a = obj.get("arguments", obj.get("parameters", {}))
            args = a if isinstance(a, str) else json.dumps(a)
        except Exception:
            pass
        self.tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": args or "{}"},
        })

    def finish(self):
        return self.feed("", final=True), self.tool_calls


# --------------------------------------------------------------------------- #
# Inference: stream text deltas out of llama.cpp
# --------------------------------------------------------------------------- #
def run_text(prompt: str, body: Dict[str, Any]):
    """Yield generated text deltas (worker thread only)."""
    temp = body.get("temperature")
    temp = DEF_TEMP if temp is None else float(temp)
    top_p = body.get("top_p")
    top_p = DEF_TOP_P if top_p is None else float(top_p)
    top_k = body.get("top_k")
    top_k = DEF_TOP_K if top_k is None else int(top_k)
    min_p = body.get("min_p")
    min_p = DEF_MIN_P if min_p is None else float(min_p)
    repeat = body.get("repetition_penalty")
    repeat = DEF_REPEAT if repeat is None else float(repeat)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or DEFAULT_MAX_TOKENS)

    for out in llm.create_completion(
        prompt, max_tokens=max_tokens, temperature=temp, top_p=top_p,
        top_k=top_k, min_p=min_p, repeat_penalty=repeat,
        stop=["<|im_end|>"], stream=True,
    ):
        piece = out["choices"][0].get("text", "")
        if piece:
            yield piece


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
    created = int(time.time())
    cid = "chatcmpl-" + uuid.uuid4().hex

    prompt = build_prompt(body)
    init_mode = "reasoning" if _starts_in_think(prompt) else "content"

    if not stream:
        def generate_all():
            parser = QwenStreamParser(initial_mode=init_mode)
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
        parser = QwenStreamParser(initial_mode=init_mode)
        q: "queue.Queue" = queue.Queue(maxsize=256)

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
                yield chunk({"reasoning_content": txt} if ev_kind == "reasoning"
                            else {"content": txt})

        tail, tool_calls = parser.finish()
        for ev_kind, txt in tail:
            yield chunk({"reasoning_content": txt} if ev_kind == "reasoning"
                        else {"content": txt})

        if errored:
            yield chunk({}, finish="stop")
        elif tool_calls:
            tc_delta = [{"index": i, **tc} for i, tc in enumerate(tool_calls)]
            yield chunk({"tool_calls": tc_delta}, finish="tool_calls")
        else:
            yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)