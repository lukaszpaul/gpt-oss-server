#!/usr/bin/env python3
"""
qwen-server.py — OpenAI-compatible server for Qwen3.6-35B-A3B on Apple Silicon
(MLX), purpose-built so GitHub Copilot's Custom Endpoint treats it like a native
model: correct tool calling, working agent/plan mode, and clean formatting
(reasoning never leaks into the answer).

WHY THIS REPLACES gpt-server.py (and why it's NOT a GGUF server)
  The old server targeted gpt-oss-20b, which (a) ran on MLX from safetensors and
  (b) spoke the "harmony" three-channel format. Qwen3.6 is different on both axes:

    - FORMAT: Qwen3.6 has no harmony. It emits optional <think>...</think>
      reasoning, then the answer, with tool calls as one or more
      <tool_call>\n{json}\n</tool_call> blocks. So all openai_harmony code is
      gone; the tokenizer's own chat template now renders the prompt, and a small
      streaming parser splits the output back into reasoning_content / content /
      tool_calls.

    - WEIGHTS: this loads an MLX build (e.g. mlx-community/Qwen3.6-35B-A3B-4bit
      or unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit). It does NOT load the GGUF Q4_K_M —
      MLX can't read llama.cpp k-quants. If you specifically need the GGUF, that
      path requires llama.cpp (or llama-cpp-python), not this file.

  What carries over from the old design: the single-thread Metal pinning (MLX
  command streams are per-thread) and the SSE producer/queue tee.

RUN
  python3 -m venv ~/mlx-env && source ~/mlx-env/bin/activate
  pip install -U mlx-lm fastapi "uvicorn[standard]"
  export QWEN_PATH="mlx-community/Qwen3.6-35B-A3B-4bit"   # HF repo id OR local dir
  python qwen-server.py                                   # serves http://localhost:8000

VS CODE (chatLanguageModels.json) — point the model's `url` at:
  http://localhost:8000/v1/chat/completions
  set "id" to match QWEN_ID below, "toolCalling": true, "thinking": true
"""

import os
import json
import time
import uuid
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get("QWEN_PATH", "mlx-community/Qwen3.6-35B-A3B-4bit")
MODEL_ID = os.environ.get("QWEN_ID", "qwen3.6-35b-a3b")
PORT = int(os.environ.get("QWEN_PORT", "8000"))
DEFAULT_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "4096"))

# Qwen3.6 thinking-mode sampling recipe. (Greedy/low-temp can drive these models
# into repetition loops; these defaults match Qwen's recommended thinking recipe.
# Clients may override any of them per request.)
DEF_TEMP = float(os.environ.get("QWEN_TEMP", "0.6"))
DEF_TOP_P = float(os.environ.get("QWEN_TOP_P", "0.95"))
DEF_TOP_K = int(os.environ.get("QWEN_TOP_K", "20"))
DEF_MIN_P = float(os.environ.get("QWEN_MIN_P", "0.0"))

# All MLX/Metal work MUST run on one consistent thread. Metal command streams are
# per-thread, so letting generation hop across Starlette's threadpool throws
# "There is no Stream(gpu, N) in current thread" and kills the response mid-stream.
# We pin every model touch to a single-worker executor; max_workers=1 also
# serializes overlapping requests, which MLX generation requires anyway.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def _load():
    print(f"[qwen-server] loading model from {MODEL_PATH} ...")
    return load(MODEL_PATH)            # (model, tokenizer) — tokenizer owns the template


model, tokenizer = GPU.submit(_load).result()
print(f"[qwen-server] ready on MLX worker thread. model id: {MODEL_ID}")

app = FastAPI()


# --------------------------------------------------------------------------- #
# Request (OpenAI messages) -> Qwen chat-template prompt string
# --------------------------------------------------------------------------- #
def _text_of(content: Any) -> str:
    """OpenAI content can be a string or a list of parts; flatten to text."""
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
    """Make OpenAI messages safe for Qwen's Jinja chat template.

    Mostly pass-through, but: flatten list-style content to text, and for
    assistant tool_calls turn the OpenAI JSON-string `arguments` into a dict
    (Qwen's template iterates the object, not a string)."""
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

    # Thinking is on by default; a client can disable it via enable_thinking,
    # reasoning_effort="none", or chat_template_kwargs.
    enable_thinking = body.get("enable_thinking")
    if enable_thinking is None:
        enable_thinking = str(body.get("reasoning_effort", "")).lower() != "none"
    extra = dict(body.get("chat_template_kwargs") or {})

    def render(**kw) -> str:
        return tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw
        )

    # Some template versions don't accept enable_thinking; degrade gracefully.
    try:
        return render(enable_thinking=enable_thinking, **extra)
    except TypeError:
        return render(**extra)


# --------------------------------------------------------------------------- #
# Streaming parser: Qwen text stream -> reasoning / content / tool_calls
# --------------------------------------------------------------------------- #
class QwenStreamParser:
    """Incrementally split a Qwen3.6 text stream. Qwen emits optional
    <think>...</think> first, then content that may contain one or more
    <tool_call>\n{json}\n</tool_call> blocks.

    feed(delta) -> list of ("reasoning"|"content", str) events to stream now.
    Tool calls accumulate; flush them with finish()."""

    _MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
    _HOLD = max(len(m) for m in _MARKERS) - 1   # hold back possible partial tag

    def __init__(self):
        self.buf = ""
        self.mode = "content"        # "content" | "reasoning"
        self.in_tool = False
        self._tool_buf = ""
        self.tool_calls: List[Dict[str, Any]] = []

    def _emit(self, text, events):
        if text:
            events.append((self.mode, text))   # mode is "reasoning" or "content"

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
# Inference: stream text deltas out of MLX
# --------------------------------------------------------------------------- #
def run_text(body: Dict[str, Any]):
    """Yield generated text deltas until EOS or max_tokens (worker thread only)."""
    prompt = build_prompt(body)

    temp = body.get("temperature")
    temp = DEF_TEMP if temp is None else float(temp)
    top_p = body.get("top_p")
    top_p = DEF_TOP_P if top_p is None else float(top_p)
    top_k = body.get("top_k")
    top_k = DEF_TOP_K if top_k is None else int(top_k)
    min_p = body.get("min_p")
    min_p = DEF_MIN_P if min_p is None else float(min_p)

    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)

    logits_processors = None
    rep = body.get("repetition_penalty") or body.get("presence_penalty")
    if rep:
        logits_processors = make_logits_processors(repetition_penalty=float(rep))

    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or DEFAULT_MAX_TOKENS)

    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=sampler, logits_processors=logits_processors,
    ):
        if resp.text:
            yield resp.text


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

    if not stream:
        def generate_all():
            parser = QwenStreamParser()
            reasoning, content = "", ""
            for delta in run_text(body):
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

    # Streaming: a producer runs generation on the MLX worker thread and pushes
    # text deltas into a queue. The SSE generator (which Starlette may iterate on
    # any threadpool thread) only drains the queue and runs the pure-Python parse.
    # No Metal op ever leaves the worker.
    def sse():
        parser = QwenStreamParser()
        q: "queue.Queue" = queue.Queue(maxsize=256)

        def produce():
            try:
                for delta in run_text(body):     # all MLX work, worker thread
                    q.put(("txt", delta))
            except Exception as e:               # surface crashes to the client
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
            for ev_kind, txt in parser.feed(val):    # pure-Python parse, safe here
                if ev_kind == "reasoning":
                    yield chunk({"reasoning_content": txt})
                else:
                    yield chunk({"content": txt})

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