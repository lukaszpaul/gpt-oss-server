#!/usr/bin/env python3
"""
ide_backend.py — a barebones local IDE backend that puts YOU in control of the
AI workflow (instead of GitHub Copilot Chat).

It serves a small HTML/JS frontend (./frontend) and exposes:

  - A file explorer for any folder you open (lazy directory listing).
  - Read / write for any file inside the opened folder.
  - A chat WebSocket with two modes:
       ASK   — read-only. The model can read files and list dirs to answer
               questions, but cannot modify anything.
       AGENT — full. The model can read AND write files inside the workspace,
               and can run git commands — but every git command requires your
               explicit approval in the UI before it runs.

The actual language model is served by ide.py (the copied repo-map server),
which exposes an OpenAI-compatible API at MODEL_BASE (default 127.0.0.1:8000).
This backend is a thin orchestrator: it runs the agent/tool loop and brokers
permission for anything that touches your machine.

RUN
  pip install fastapi "uvicorn[standard]" httpx
  # terminal 1: the model (Apple Silicon / MLX)
  python ide.py
  # terminal 2: this backend
  python ide_backend.py            # http://127.0.0.1:8001
  open http://127.0.0.1:8001

CONFIG (env)
  IDE_PORT          backend port (default 8001)
  IDE_MODEL_BASE    model server base URL (default http://127.0.0.1:8000)
  IDE_MAX_STEPS     max agent tool-loop iterations per message (default 12)
  IDE_GIT_TIMEOUT   seconds before a git command is killed (default 30)
"""

import os
import json
import uuid
import shlex
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"
SYSTEM_PROMPT_FILE = HERE / "system_prompt.md"

PORT = int(os.environ.get("IDE_PORT", "8001"))
MODEL_BASE = os.environ.get("IDE_MODEL_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL_CHAT_URL = MODEL_BASE + "/v1/chat/completions"
MODEL_LIST_URL = MODEL_BASE + "/v1/models"
MAX_STEPS = int(os.environ.get("IDE_MAX_STEPS", "12"))
GIT_TIMEOUT = int(os.environ.get("IDE_GIT_TIMEOUT", "30"))

# Files/dirs we hide from the tree by default (still openable if you ask the
# model for them by path — this is only the explorer's noise filter).
HIDE = {".git", "__pycache__", ".DS_Store", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", ".idea", "captures", "repomaps"}

# Read cap so the model can't blow its own context on a huge file.
MAX_READ_BYTES = 200_000


# --------------------------------------------------------------------------- #
# Workspace state (single-user local tool: one open folder at a time)
# --------------------------------------------------------------------------- #
class Workspace:
    def __init__(self) -> None:
        self.root: Optional[Path] = None

    def open(self, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise ValueError(f"Not a directory: {p}")
        self.root = p
        return p

    def resolve(self, rel: str) -> Path:
        """Resolve a (possibly relative) path and confine it to the root."""
        if self.root is None:
            raise ValueError("No workspace folder is open.")
        rel = (rel or "").strip()
        cand = Path(rel)
        if not cand.is_absolute():
            cand = self.root / cand
        cand = cand.resolve()
        # Confinement: cand must be the root or inside it.
        if cand != self.root and self.root not in cand.parents:
            raise ValueError(f"Path escapes the workspace: {rel}")
        return cand

    def rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.root)).replace("\\", "/")
        except Exception:
            return str(p)


WS = Workspace()


def list_dir(root_rel: str = "") -> List[Dict[str, Any]]:
    base = WS.resolve(root_rel) if root_rel else WS.root
    if base is None:
        raise ValueError("No workspace folder is open.")
    if not base.is_dir():
        raise ValueError(f"Not a directory: {WS.rel(base)}")
    entries = []
    for child in sorted(base.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
        if child.name in HIDE:
            continue
        entries.append({
            "name": child.name,
            "path": WS.rel(child),
            "type": "dir" if child.is_dir() else "file",
        })
    return entries


# --------------------------------------------------------------------------- #
# Tool implementations (what the model can actually do)
# --------------------------------------------------------------------------- #
def tool_read_file(args: Dict[str, Any]) -> str:
    p = WS.resolve(args.get("path", ""))
    if not p.is_file():
        return f"ERROR: not a file: {args.get('path')}"
    data = p.read_bytes()[:MAX_READ_BYTES]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {WS.rel(p)} is not UTF-8 text (binary file)."
    note = "" if len(p.read_bytes()) <= MAX_READ_BYTES else \
        f"\n\n[...truncated at {MAX_READ_BYTES} bytes]"
    return f"--- {WS.rel(p)} ---\n{text}{note}"


def tool_list_dir(args: Dict[str, Any]) -> str:
    entries = list_dir(args.get("path", ""))
    lines = [("[dir]  " if e["type"] == "dir" else "       ") + e["path"]
             for e in entries]
    head = args.get("path") or "."
    return f"Contents of {head}:\n" + ("\n".join(lines) if lines else "(empty)")


def tool_write_file(args: Dict[str, Any]) -> str:
    p = WS.resolve(args.get("path", ""))
    content = args.get("content")
    if content is None:
        return "ERROR: write_file requires 'content'."
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {WS.rel(p)} ({len(content)} chars)."


def run_git(arg_string: str) -> str:
    if WS.root is None:
        return "ERROR: no workspace open."
    try:
        parts = shlex.split(arg_string)
    except ValueError as e:
        return f"ERROR: could not parse git args: {e}"
    if parts and parts[0] == "git":
        parts = parts[1:]
    try:
        proc = subprocess.run(
            ["git", *parts],
            cwd=str(WS.root),
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: `git {arg_string}` timed out after {GIT_TIMEOUT}s."
    except FileNotFoundError:
        return "ERROR: git is not installed or not on PATH."
    out = (proc.stdout or "") + (proc.stderr or "")
    out = out.strip() or "(no output)"
    if len(out) > 20_000:
        out = out[:20_000] + "\n[...truncated]"
    return f"$ git {arg_string}\n(exit {proc.returncode})\n{out}"


# Tool schemas advertised to the model. ASK gets the read-only subset.
def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required},
    }}


TOOL_READ = _schema(
    "read_file", "Read a UTF-8 text file inside the workspace and return its "
    "contents. Use this to inspect code before answering or editing.",
    {"path": {"type": "string",
              "description": "Path relative to the workspace root."}},
    ["path"])

TOOL_LIST = _schema(
    "list_dir", "List the files and subdirectories of a directory inside the "
    "workspace.",
    {"path": {"type": "string",
              "description": "Directory path relative to the workspace root. "
                             "Empty string means the workspace root."}},
    [])

TOOL_WRITE = _schema(
    "write_file", "Create or overwrite a text file inside the workspace with "
    "the given full contents. Always pass the COMPLETE new file content.",
    {"path": {"type": "string", "description": "Path relative to the root."},
     "content": {"type": "string", "description": "Full new file contents."}},
    ["path", "content"])

TOOL_GIT = _schema(
    "run_git", "Run a git command in the workspace root. The user must approve "
    "each invocation before it runs. Pass only the arguments after 'git', e.g. "
    "'status', 'add -A', 'commit -m \"msg\"', 'diff HEAD'.",
    {"args": {"type": "string",
              "description": "Git arguments, e.g. 'status' or 'log --oneline -n 5'."}},
    ["args"])

TOOLS_ASK = [TOOL_READ, TOOL_LIST]
TOOLS_AGENT = [TOOL_READ, TOOL_LIST, TOOL_WRITE, TOOL_GIT]


# --------------------------------------------------------------------------- #
# System prompt assembly
# --------------------------------------------------------------------------- #
def build_system(mode: str) -> str:
    base = ""
    if SYSTEM_PROMPT_FILE.exists():
        base = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    root = str(WS.root) if WS.root else "(none open)"
    # The workspace_info block below is parsed by ide.py's repo-map injector,
    # which hands the model a free structural map of the opened folder.
    workspace_info = (
        "I am working in a workspace with the following folders:\n"
        f"- {root}\n"
    )
    if mode == "agent":
        cap = (
            "MODE: AGENT. You may read and write files inside the workspace, "
            "and you may run git commands (each git command is shown to the "
            "user for approval before it runs — if denied, adapt). Make changes "
            "directly with write_file; do not just describe them. After editing, "
            "briefly summarize what you changed."
        )
    else:
        cap = (
            "MODE: ASK. You are read-only. You may read files and list "
            "directories to answer the user's question, but you must NOT modify "
            "any file and you have no write or git tools. If the answer requires "
            "a change, describe the change instead of making it."
        )
    return f"{base}\n\n{workspace_info}\n{cap}"


# --------------------------------------------------------------------------- #
# Model client + streaming agent loop
# --------------------------------------------------------------------------- #
MODEL_ID = "local"


async def fetch_model_id() -> None:
    global MODEL_ID
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(MODEL_LIST_URL)
            data = r.json()
            MODEL_ID = data["data"][0]["id"]
    except Exception:
        MODEL_ID = "local"


class Session:
    """Per-WebSocket conversation + pending permission futures."""
    def __init__(self) -> None:
        self.conversation: List[Dict[str, Any]] = []
        self.pending: Dict[str, asyncio.Future] = {}


async def stream_model(ws: WebSocket, body: Dict[str, Any]):
    """POST to the model with stream=True, forward reasoning/content deltas to
    the browser, and return (reasoning, content, tool_calls)."""
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    tool_calls: Dict[int, Dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", MODEL_CHAT_URL, json=body) as resp:
            if resp.status_code != 200:
                txt = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"model {resp.status_code}: {txt[:500]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                    await ws.send_json({"type": "reasoning",
                                        "text": delta["reasoning_content"]})
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    await ws.send_json({"type": "content",
                                        "text": delta["content"]})
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(
                        idx, {"id": tc.get("id") or f"call_{idx}",
                              "type": "function",
                              "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

    calls = [tool_calls[i] for i in sorted(tool_calls)]
    return "".join(reasoning_parts), "".join(content_parts), calls


async def request_permission(ws: WebSocket, sess: Session, command: str) -> bool:
    pid = uuid.uuid4().hex
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    sess.pending[pid] = fut
    await ws.send_json({"type": "permission_request", "id": pid,
                        "command": command})
    try:
        return await fut
    finally:
        sess.pending.pop(pid, None)


async def exec_tool(ws: WebSocket, sess: Session, tc: Dict[str, Any],
                    mode: str) -> str:
    name = tc["function"]["name"]
    raw_args = tc["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if raw_args.strip() else {}
    except json.JSONDecodeError:
        args = {}

    await ws.send_json({"type": "tool_start", "id": tc["id"],
                        "name": name, "args": args})

    try:
        if name == "read_file":
            result = await asyncio.to_thread(tool_read_file, args)
            status = "ok"
        elif name == "list_dir":
            result = await asyncio.to_thread(tool_list_dir, args)
            status = "ok"
        elif name == "write_file":
            if mode != "agent":
                result, status = "ERROR: write_file is disabled in ASK mode.", "error"
            else:
                result = await asyncio.to_thread(tool_write_file, args)
                status = "ok"
        elif name == "run_git":
            if mode != "agent":
                result, status = "ERROR: run_git is disabled in ASK mode.", "error"
            else:
                cmd = "git " + str(args.get("args", "")).strip()
                approved = await request_permission(ws, sess, cmd)
                if not approved:
                    result, status = f"DENIED by user: {cmd}", "denied"
                else:
                    result = await asyncio.to_thread(run_git, args.get("args", ""))
                    status = "ok"
        else:
            result, status = f"ERROR: unknown tool '{name}'.", "error"
    except Exception as e:  # tool errors are fed back to the model, not fatal
        result, status = f"ERROR: {type(e).__name__}: {e}", "error"

    await ws.send_json({"type": "tool_result", "id": tc["id"],
                        "name": name, "status": status, "result": result})
    return result


async def run_agent(ws: WebSocket, sess: Session, data: Dict[str, Any]) -> None:
    mode = "agent" if data.get("mode") == "agent" else "ask"
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        await ws.send_json({"type": "done"})
        return
    if WS.root is None:
        await ws.send_json({"type": "error",
                            "message": "Open a folder first."})
        await ws.send_json({"type": "done"})
        return

    sess.conversation.append({"role": "user", "content": user_msg})
    tools = TOOLS_AGENT if mode == "agent" else TOOLS_ASK

    try:
        for _ in range(MAX_STEPS):
            messages = [{"role": "system", "content": build_system(mode)}]
            messages += sess.conversation
            body = {"model": MODEL_ID, "messages": messages,
                    "tools": tools, "stream": True}

            reasoning, content, tool_calls = await stream_model(ws, body)

            asst: Dict[str, Any] = {"role": "assistant",
                                    "content": content or None}
            if reasoning.strip():
                asst["reasoning_content"] = reasoning
            if tool_calls:
                asst["tool_calls"] = tool_calls
            sess.conversation.append(asst)

            if not tool_calls:
                break

            for tc in tool_calls:
                result = await exec_tool(ws, sess, tc, mode)
                sess.conversation.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["function"]["name"], "content": result})
        else:
            await ws.send_json({"type": "content",
                                "text": f"\n\n[stopped after {MAX_STEPS} steps]"})
    except asyncio.CancelledError:
        await ws.send_json({"type": "content", "text": "\n\n[cancelled]"})
        raise
    except Exception as e:
        await ws.send_json({"type": "error",
                            "message": f"{type(e).__name__}: {e}"})

    await ws.send_json({"type": "done"})


# --------------------------------------------------------------------------- #
# HTTP + WebSocket app
# --------------------------------------------------------------------------- #
app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    await fetch_model_id()


@app.get("/")
async def index() -> Response:
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            ok = (await c.get(MODEL_LIST_URL)).status_code == 200
    except Exception:
        ok = False
    return {"model_base": MODEL_BASE, "model_id": MODEL_ID,
            "model_up": ok, "root": str(WS.root) if WS.root else None}


@app.post("/api/open")
async def api_open(request: Request) -> Response:
    body = await request.json()
    try:
        root = WS.open(body.get("path", ""))
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return JSONResponse({"root": str(root), "name": root.name,
                         "entries": list_dir("")})


@app.get("/api/dir")
async def api_dir(path: str = "") -> Response:
    try:
        return JSONResponse({"entries": list_dir(path)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/file")
async def api_file(path: str) -> Response:
    try:
        p = WS.resolve(path)
        if not p.is_file():
            return JSONResponse(status_code=404,
                                content={"error": "not a file"})
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return JSONResponse(status_code=415,
                                content={"error": "binary / non-UTF-8 file"})
        return JSONResponse({"path": WS.rel(p), "content": content})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.put("/api/file")
async def api_file_write(request: Request) -> Response:
    body = await request.json()
    try:
        p = WS.resolve(body.get("path", ""))
        content = body.get("content", "")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return JSONResponse({"ok": True, "path": WS.rel(p)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.websocket("/ws")
async def ws_chat(ws: WebSocket) -> None:
    await ws.accept()
    sess = Session()
    agent_task: Optional[asyncio.Task] = None
    try:
        while True:
            data = await ws.receive_json()
            kind = data.get("type")
            if kind == "chat":
                if agent_task and not agent_task.done():
                    await ws.send_json({"type": "error",
                                        "message": "Still working — wait or cancel."})
                    continue
                agent_task = asyncio.create_task(run_agent(ws, sess, data))
            elif kind == "permission":
                fut = sess.pending.get(data.get("id", ""))
                if fut and not fut.done():
                    fut.set_result(bool(data.get("approved")))
            elif kind == "cancel":
                if agent_task and not agent_task.done():
                    agent_task.cancel()
            elif kind == "reset":
                sess.conversation.clear()
                await ws.send_json({"type": "reset_ok"})
    except WebSocketDisconnect:
        if agent_task and not agent_task.done():
            agent_task.cancel()


# Serve the frontend at the web root so index.html's relative references
# (style.css, app.js) resolve. Mounted LAST, after the /api and /ws routes,
# which are registered earlier and therefore match first. html=True makes "/"
# return index.html. Also kept under /static for backwards compatibility.
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="root")


if __name__ == "__main__":
    print(f"[ide_backend] serving http://127.0.0.1:{PORT}  (model: {MODEL_BASE})")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
