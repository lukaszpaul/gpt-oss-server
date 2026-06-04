#!/usr/bin/env python3
"""
gpt-server.py — OpenAI-compatible server for gpt-oss-20b on Apple Silicon (MLX),
purpose-built so GitHub Copilot's Custom Endpoint treats it like a native model:
correct tool calling, working agent/plan mode, and clean formatting (reasoning
never leaks into the answer).

WHY THIS EXISTS
  gpt-oss speaks the "harmony" response format with three channels:
    - analysis    -> chain-of-thought (must NOT appear in the final answer)
    - commentary  -> where the model emits tool/function calls
    - final       -> the user-facing answer
  Copilot speaks the OpenAI Chat Completions API. This server is the translator:
  it renders incoming requests INTO harmony, runs MLX inference, then parses the
  harmony output BACK into OpenAI shape (reasoning_content, tool_calls, content).

RUN
  python3 -m venv ~/mlx-env && source ~/mlx-env/bin/activate
  pip install -U mlx-lm openai-harmony fastapi "uvicorn[standard]"
  export GPT_OSS_PATH="$HOME/models/gpt-oss-20b"     # folder w/ config.json + *.safetensors
  python gpt-server.py                                # serves http://localhost:8000

VS CODE (chatLanguageModels.json) — point the model's `url` at:
  http://localhost:8000/v1/chat/completions
  set "id": "gpt-oss-20b", "toolCalling": true, "thinking": true
"""

import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    Author,
    SystemContent,
    DeveloperContent,
    ToolDescription,
    ReasoningEffort,
    StreamableParser,
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
MODEL_PATH = os.environ.get("GPT_OSS_PATH", os.path.expanduser("~/models/gpt-oss-20b"))
MODEL_ID = os.environ.get("GPT_OSS_ID", "gpt-oss-20b")
PORT = int(os.environ.get("GPT_OSS_PORT", "8000"))
DEFAULT_MAX_TOKENS = int(os.environ.get("GPT_OSS_MAX_TOKENS", "4096"))

print(f"[gpt-server] loading model from {MODEL_PATH} ...")
model, _tok = load(MODEL_PATH)                       # _tok unused: harmony owns tokenization
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
STOP_IDS = set(enc.stop_tokens_for_assistant_actions())
print(f"[gpt-server] ready. harmony stop tokens: {sorted(STOP_IDS)}")

_EFFORT = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}

app = FastAPI()


# --------------------------------------------------------------------------- #
# Request (OpenAI messages) -> harmony Conversation
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


def build_conversation(body: Dict[str, Any]) -> Conversation:
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    effort = _EFFORT.get(str(body.get("reasoning_effort", "medium")).lower(),
                         ReasoningEffort.MEDIUM)

    # System channel: reserved harmony metadata (reasoning effort, identity).
    sys_content = SystemContent.new().with_reasoning_effort(effort)

    # Developer channel: the user's "system" prompt(s) + the tool definitions.
    instructions = "\n\n".join(
        _text_of(m.get("content")) for m in messages if m.get("role") == "system"
    ).strip()

    dev = DeveloperContent.new()
    if instructions:
        dev = dev.with_instructions(instructions)
    if tools:
        tool_descs = []
        for t in tools:
            fn = t.get("function", t)
            tool_descs.append(
                ToolDescription.new(
                    fn.get("name", ""),
                    fn.get("description", ""),
                    fn.get("parameters"),     # JSON-schema dict or None
                )
            )
        dev = dev.with_function_tools(tool_descs)

    convo: List[Message] = [
        Message.from_role_and_content(Role.SYSTEM, sys_content),
        Message.from_role_and_content(Role.DEVELOPER, dev),
    ]

    # Map tool_call_id -> function name so tool results can be authored correctly.
    id_to_name: Dict[str, str] = {}

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue

        if role == "user":
            convo.append(Message.from_role_and_content(Role.USER, _text_of(m.get("content"))))

        elif role == "assistant":
            # Replay any prior tool calls on the commentary channel.
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "") or ""
                id_to_name[tc.get("id", "")] = name
                convo.append(
                    Message.from_role_and_content(Role.ASSISTANT, args)
                    .with_channel("commentary")
                    .with_recipient(f"functions.{name}")
                    .with_content_type("json")
                )
            text = _text_of(m.get("content"))
            if text:
                convo.append(
                    Message.from_role_and_content(Role.ASSISTANT, text).with_channel("final")
                )

        elif role == "tool":
            # Tool result -> authored by the function, addressed back to assistant.
            name = id_to_name.get(m.get("tool_call_id", ""), m.get("name", "tool"))
            convo.append(
                Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{name}"), _text_of(m.get("content"))
                )
                .with_recipient("assistant")
                .with_channel("commentary")
            )

    return Conversation.from_messages(convo)


# --------------------------------------------------------------------------- #
# Inference: stream harmony tokens out of MLX
# --------------------------------------------------------------------------- #
def run_tokens(body: Dict[str, Any]):
    """Yield generated token ids until a harmony stop token or max_tokens."""
    convo = build_conversation(body)
    prompt_ids = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

    temp = float(body.get("temperature", 1.0) or 0.0)
    top_p = float(body.get("top_p", 1.0) or 0.0)
    sampler = make_sampler(temp=temp, top_p=top_p)

    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                     or DEFAULT_MAX_TOKENS)

    prompt = mx.array(prompt_ids)
    n = 0
    for token, _logprobs in generate_step(prompt, model, max_tokens=-1, sampler=sampler):
        if token in STOP_IDS:
            break
        yield token
        n += 1
        if n >= max_tokens:
            break


# --------------------------------------------------------------------------- #
# Parse harmony messages -> OpenAI response pieces
# --------------------------------------------------------------------------- #
def parse_final(tokens: List[int]):
    """Non-streaming: split tokens into reasoning, content, tool_calls."""
    msgs = enc.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
    reasoning, content, tool_calls = "", "", []
    for msg in msgs:
        d = msg.to_dict()
        channel = d.get("channel")
        recipient = d.get("recipient")
        text = "".join(
            c.get("text", "") for c in d.get("content", []) if isinstance(c, dict)
        )
        if recipient and str(recipient).startswith("functions."):
            tool_calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": str(recipient).split("functions.", 1)[1],
                             "arguments": text or "{}"},
            })
        elif channel == "analysis":
            reasoning += text
        elif channel == "final":
            content += text
        else:
            content += text
    return reasoning, content, tool_calls


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
        tokens = list(run_tokens(body))
        reasoning, content, tool_calls = parse_final(tokens)
        message: Dict[str, Any] = {"role": "assistant",
                                   "content": content if content else None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        })

    # Streaming: emit reasoning_content + content deltas live; tool calls are
    # consolidated into one delta at the end (Copilot accepts this), with the
    # correct finish_reason so the agent loop triggers.
    def sse():
        parser = StreamableParser(enc, role=Role.ASSISTANT)
        tool_calls: List[Dict[str, Any]] = []
        cur_tool: Optional[Dict[str, Any]] = None

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
            return "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }) + "\n\n"

        yield chunk({"role": "assistant"})

        for token in run_tokens(body):
            parser.process(token)
            delta = parser.last_content_delta
            if not delta:
                continue
            channel = parser.current_channel
            recipient = parser.current_recipient

            if recipient and str(recipient).startswith("functions."):
                name = str(recipient).split("functions.", 1)[1]
                if cur_tool is None or cur_tool["function"]["name"] != name:
                    cur_tool = {"id": "call_" + uuid.uuid4().hex[:24],
                                "type": "function",
                                "function": {"name": name, "arguments": ""}}
                    tool_calls.append(cur_tool)
                cur_tool["function"]["arguments"] += delta
            elif channel == "analysis":
                yield chunk({"reasoning_content": delta})
            else:                                    # "final" (or stray) -> content
                yield chunk({"content": delta})

        if tool_calls:
            tc_delta = [{"index": i, **tc} for i, tc in enumerate(tool_calls)]
            yield chunk({"tool_calls": tc_delta}, finish="tool_calls")
        else:
            yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)