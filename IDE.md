# Local IDE (you-control-the-AI)

A barebones IDE that replaces GitHub Copilot Chat with **your** local model and
**your** rules. Open any folder, browse and edit its files, and chat with the
model in one of two modes — while you stay in control of everything that touches
your machine.

```
┌──────────── browser (http://127.0.0.1:8001) ────────────┐
│  Explorer  │      Editor       │        Chat            │
│  (tree)    │   (read / save)   │   ASK / AGENT toggle   │
└─────────────────────────┬──────────────────────────────┘
                          │ WebSocket + REST
                ┌─────────▼──────────┐
                │  ide_backend.py    │  file ops, git (with approval),
                │  (FastAPI :8001)   │  agent/tool loop
                └─────────┬──────────┘
                          │ OpenAI-compatible HTTP
                ┌─────────▼──────────┐
                │  ide.py (:8000)    │  the model (MLX, Apple Silicon)
                │  = repo-map server │  copied from gpt-server-27b-repomap.py
                └────────────────────┘
```

## Pieces

| File | Role |
|------|------|
| `ide.py` | The AI engine — a copy of the repo-map MLX server. Serves the model on `:8000`. |
| `ide_backend.py` | The IDE backend (FastAPI, `:8001`). File tree/read/write, git-with-permission, agent loop. |
| `frontend/` | The UI (`index.html`, `style.css`, `app.js`). Dependency-free, runs offline. |
| `system_prompt.md` | The system prompt. Edited live — re-read on every message, no restart. |

## Run

```bash
pip install fastapi "uvicorn[standard]" httpx

# Terminal 1 — the model (on the Mac / Apple Silicon)
python ide.py                      # http://127.0.0.1:8000

# Terminal 2 — the IDE backend
python ide_backend.py              # http://127.0.0.1:8001
```

Open <http://127.0.0.1:8001>, type a folder path in the top bar (e.g.
`/Users/you/code/project`), and hit **Open folder**.

## Modes

- **ASK** — read-only. The model can `read_file` and `list_dir` to answer
  questions, but has no write or git tools. Nothing on disk changes.
- **AGENT** — full. The model can `read_file`, `list_dir`, `write_file`
  (anywhere inside the open folder), and `run_git`. **Every git command pops an
  Approve / Deny prompt in the chat** before it runs.

## Safety model

- All file paths are confined to the opened folder — `..` and absolute paths
  that escape the root are rejected (returns HTTP 400 / tool error).
- Writes happen only in AGENT mode; ASK mode refuses them.
- Git never runs without an explicit click. Denials are fed back to the model so
  it can adapt.
- The model itself stays local — nothing leaves your machine.

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `IDE_PORT` | `8001` | Backend port. |
| `IDE_MODEL_BASE` | `http://127.0.0.1:8000` | Where `ide.py` is serving. |
| `IDE_MAX_STEPS` | `12` | Max tool-loop iterations per message. |
| `IDE_GIT_TIMEOUT` | `30` | Seconds before a git command is killed. |
| `QWEN_PORT` | `8000` | Port `ide.py` listens on (from the base server). |

## Notes / next steps

- The editor is a plain textarea (no syntax highlighting yet) — deliberately
  barebones. CodeMirror/Monaco can be dropped in later.
- `ide.py` auto-injects a **repo map** of the opened folder for free, because the
  backend passes a `workspace_info` block the server already knows how to parse —
  so AGENT mode gets oriented without burning tool calls.
- Conversation history is per-WebSocket and in-memory; **Reset** clears it.
