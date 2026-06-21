# Server Controls

Every knob for `gpt-server-27b-repomap.py`, by category. All are environment
variables read once at startup. **Bold defaults** are what you get with nothing
set.

Legend for the **Affects** column: 🚀 = speed/TTFT, 🧠 = memory, 🎯 = answer
quality/relevance, 🔁 = cache reuse, 🐛 = diagnostics.

---

## TL;DR — the dials that actually matter

| Want to… | Set |
|---|---|
| Make **returning to an old chat** fast | `QWEN_DISK_CACHE=1` (after the round-trip test passes) |
| Make **brand-new chats** fast | `QWEN_FRONT_SNAPSHOT=1` (after the round-trip test passes) |
| Keep more chats warm in RAM (cheap, no test needed) | `QWEN_CACHE_SLOTS=5` |
| Stop RAM creeping up over a long session | `QWEN_CACHE_LIMIT_GB=6` |
| Hard cap RAM so it never OOMs | `QWEN_MEM_BUDGET_GB=40` |
| Find what's breaking cache reuse | watch `QWEN_CACHE_DEBUG=1` logs (on by default) |
| Turn off context injection entirely | `QWEN_REPOMAP=0` |

> ⚠️ `QWEN_DISK_CACHE` and `QWEN_FRONT_SNAPSHOT` are **OFF by default** and must
> not be enabled until `python test_cache_roundtrip.py` prints PASS on the
> target machine. A bad cache save/load corrupts context silently.

---

## Model & server

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_MLX` | path or HF repo | **`mlx-community/Qwen3.6-27B-OptiQ-4bit`** | local dir for fastest load | Which model weights to load. |
| `QWEN_ID` | string | **`qwen3.6-27b`** | default | Model id reported on `/v1/models` and in logs. |
| `QWEN_PORT` | 1–65535 | **`8000`** | default | HTTP port. Must match your VS Code endpoint URL. |
| `QWEN_MAX_TOKENS` | int | **`4096`** | 4096–8192 | Default max output tokens when the request doesn't specify. 🧠 |
| `QWEN_PREFILL_STEP` | int | **`2048`** | try `4096` on M4 Max | Tokens pushed through Metal per prefill step. Bigger can raise prefill t/s. 🚀 |

---

## Prompt cache & snapshot (the reuse machinery)

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_CACHE_SLOTS` | int ≥1 | **`3`** | `5–6` if you bounce between chats | Independent in-RAM prompt caches. More = more conversations stay warm without evicting each other. 🔁🧠 |
| `QWEN_SNAPSHOT_MARKER` | string / `""` | **`<context>`** | default | Boundary for the per-conversation snapshot: everything before the LAST marker is cached, the volatile tail after it recomputes each turn. `""` disables → falls back to holdback. 🔁 |
| `QWEN_SNAPSHOT_HOLDBACK` | int | **`64`** | default | Fallback: cache all but the last N tokens when the marker isn't found (e.g. side-requests). 🔁 |
| `QWEN_KV_BITS` | `0,4,8` | **`8`** | `8` (or `4` if 128k contexts squeeze) | KV-cache quantization for the growing attention layers. Lower = less memory, slight quality risk. 🧠🎯 |
| `QWEN_KV_GROUP` | int | **`64`** | default | Group size for KV quantization. 🧠 |
| `QWEN_SORT_TOOLS` | `0/1` | **`1`** | `1` | Sort the tools array by name so the rendered prompt is deterministic. Off → a shuffled tool list breaks ALL prefix reuse. 🔁 |
| `QWEN_PIN` | JSON `[[regex,repl],…]` | **`""`** | pin volatile dates/session-ids if the debug log shows churn | Regex substitutions on the rendered prompt to neutralize volatile substrings that break reuse. The model SEES the pinned value. 🔁 |
| `QWEN_WIRED_GB` | float GB | **`0`** (off) | `56` on 64GB (needs sysctl) | Ask MLX to wire its GPU buffers so they resist page eviction between turns. 🚀🧠 |

---

## Disk-backed snapshot store ⭐ (new — your conversation-reuse fix)

> Needs `test_cache_roundtrip.py` to PASS first. Either feature below activates
> the on-disk store; they share the same directory and reload path.

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_DISK_CACHE` | `0/1` | **`0`** | `1` after test | Spill an evicted conversation's snapshot to SSD instead of discarding it; reload on return (disk read + tiny prefill vs full reprefill). Fixes **returning to an old chat**. 🚀🔁 |
| `QWEN_FRONT_SNAPSHOT` | `0/1` | **`0`** | `1` after test | Persist the shared front (system+tools+repoMap, up to the FIRST `<context>`) so a **brand-new chat** warm-loads it instead of reprefilling. Works independently of `QWEN_DISK_CACHE`. 🚀🔁 |
| `QWEN_DISK_CACHE_DIR` | path | **`./cachesnaps`** | fast SSD path | Where snapshots (`.safetensors` + sidecar `.json`) live. 🧠 |
| `QWEN_DISK_CACHE_MAX_GB` | float / `0` | **`100`** | size to taste; `0` = no cap | Total disk budget; oldest snapshots evicted past it. |
| `QWEN_DISK_CACHE_MIN_TOKENS` | int | **`2048`** | default | Don't bother spilling snapshots shorter than this — reprefill is already cheap. |

**A/B matrix:**

| `DISK_CACHE` | `FRONT_SNAPSHOT` | Behavior |
|:---:|:---:|---|
| 0 | 0 | Nothing persists (stock behavior) |
| 0 | 1 | Only fronts → new chats fast, returning chats still cold |
| 1 | 0 | Returning chats fast, new chats cold |
| 1 | 1 | Both fast |

---

## Memory governor (all OFF unless set)

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_MEM_BUDGET_GB` | float / `0` | **`0`** (off) | `40` on 64GB | Soft RAM ceiling. Evicts LRU slots before prefill to fit; refuses (503) if it still can't. 🧠 |
| `QWEN_MEM_STRICT` | `0/1` | **`1`** | `1` | With budget on: strict refuses an over-budget request; `0` warns and tries anyway. 🧠 |
| `QWEN_CACHE_LIMIT_GB` | float / `0` | **`0`** (off) | `4–8` | Cap MLX's reusable buffer pool so freed KV returns to the OS. Fixes "RSS creeps up each request". 🧠 |
| `QWEN_MAX_CONTEXT` | int / `0` | **`0`** (off) | `0` or your token ceiling | Refuse prompts longer than this (clean error vs OOM). 🧠 |
| `QWEN_KV_BYTES_PER_TOK` | int | **`70000`** | default | Seed for the self-calibrating per-token KV estimate the governor uses. Auto-refined after each prefill, so only matters for request #1. 🧠 |
| `QWEN_MEM_LOG` | `0/1` | **`1`** | `1` | Log MLX allocator stats (active/cache/peak) + RSS at each request phase. 🐛 |

---

## Sampling (coding-leaning defaults)

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_THINKING` | `0/1` | **`1`** | `1` | Global `<think>` reasoning on/off. Per-request `reasoning_effort`/`enable_thinking` override it. 🎯 |
| `QWEN_TEMP` | float | **`0.7`** | 0.6–0.7 coding | Sampling temperature. Lower = more deterministic. 🎯 |
| `QWEN_TOP_P` | 0–1 | **`0.8`** | default | Nucleus sampling cutoff. 🎯 |
| `QWEN_TOP_K` | int | **`20`** | default | Top-k sampling cutoff. 🎯 |
| `QWEN_MIN_P` | 0–1 | **`0.0`** | default | Min-p floor (off at 0). 🎯 |
| `QWEN_REPEAT_PENALTY` | float | **`1.05`** | default | Multiplicative repetition penalty. 🎯 |
| `QWEN_PRESENCE_PENALTY` | float | **`1.5`** | default | Additive presence penalty (Qwen3.6's recommended anti-loop knob). 🎯 |

---

## Chat logging

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_CHAT_LOG` | `0/1` | **`1`** | `1` while tuning | Write one JSON per request (messages, rendered prompt, response). 🐛 |
| `QWEN_CHAT_LOG_DIR` | path | **`./chats`** | default | Where those logs go. 🐛 |
| `QWEN_CACHE_DEBUG` | `0/1` | **`1`** | `1` | On a cache miss, log WHERE the prompt diverged from the slot snapshot, with text on both sides. **This is the tool that shows your reuse %.** 🐛🔁 |

---

## Repo map / context injection (ON by default)

Drives the `<repoMap>` static map and `<relevantFiles>` dynamic hint injected
into every request. See the file header for the full design.

| Knob | Values | Default | Suggested | Effect |
|---|---|---|---|---|
| `QWEN_REPOMAP` | `0/1` | **`1`** | `1` | Master switch for all context injection. Off → model explores via tool calls. 🎯🚀 |
| `QWEN_REPOMAP_DIR` | path | **`./repomaps`** | default | On-disk cache of built indexes (one JSON per workspace root). |
| `QWEN_REPOMAP_TTL` | seconds | **`300`** | 300 | How long an index is trusted before re-checking git HEAD. Within it, served from memory (byte-stable → keeps the KV snapshot valid). 🔁 |
| `QWEN_REPOMAP_MAX_CHARS` | int | **`16000`** | raise on big repos | Char budget for the static detailed section. Snapshot-cached, so nearly free after turn 1. 🎯🧠 |
| `QWEN_REPOMAP_MAX_FILES` | int | **`8000`** | default | Stop walking after this many files (guards huge trees). 🚀 |
| `QWEN_REPOMAP_MAX_FILE_BYTES` | int | **`200000`** | default | Per-file byte cap for symbol scanning (head of file only). 🚀 |
| `QWEN_REPOMAP_SYMBOLS` | `0/1` | **`1`** | `1` | Extract top-level symbols per file. This is what answers "where does X live". 🎯 |
| `QWEN_REPOMAP_SYMS_PER_FILE` | int | **`12`** | default | Max symbols listed per file (stops fat files dominating). 🎯🧠 |
| `QWEN_REPOMAP_BREADTH` | `0/1` | **`1`** | `1` | Add a collapsed per-directory summary for files not shown in detail (whole-repo shape). 🎯 |
| `QWEN_REPOMAP_BREADTH_MAX_CHARS` | int | **`8000`** | default | Char budget for that breadth summary. 🧠 |
| `QWEN_REPOMAP_GRAPH` | `0/1` | **`1`** | `1` | PageRank a symbol-reference graph so central (widely-used) files rank higher. 🎯 |
| `QWEN_REPOMAP_GRAPH_BOOST` | float | **`2.0`** | default | How much centrality boosts a file's rank on top of its role weight. 🎯 |
| `QWEN_REPOMAP_GRAPH_MAX_FILES` | int | **`6000`** | default | Skip the graph above this file count (it gets slow); fall back to role weight. 🚀 |
| `QWEN_REPOMAP_GRAPH_READ_BYTES` | int | **`65536`** | default | Bytes scanned per file when building reference edges. 🚀 |
| `QWEN_REPOMAP_HINT` | `0/1` | **`1`** | `1` | Inject `<relevantFiles>` — deterministic per-question retrieval (the "agent in the middle"). Replaces the ~100k-token investigation. 🎯🚀 |
| `QWEN_REPOMAP_HINT_FILES` | int | **`12`** | default | How many files the hint lists. 🎯 |
| `QWEN_REPOMAP_TREE_SITTER` | `0/1` | **`0`** (off) | `0` (regex is the tested path) | Use tree-sitter parse trees for symbols instead of regex. Experimental; falls back to regex on any failure. Needs `pip install tree_sitter_languages`. 🎯 |
| `QWEN_REPOMAP_IGNORE` | comma list | **`""`** | add build/vendor dirs | Extra directory names to skip, on top of the built-in ignore set. 🚀 |

---

## Round-trip test (`test_cache_roundtrip.py`) — env

Run **before** enabling the disk store. Match the server's model + KV settings.

| Knob | Default | Effect |
|---|---|---|
| `QWEN_MLX`, `QWEN_KV_BITS`, `QWEN_KV_GROUP`, `QWEN_PREFILL_STEP` | as above | Must match the server so the test exercises the real cache layout. |
| `ROUNDTRIP_N_GEN` | `40` | Tokens compared token-for-token between resident and reloaded cache. |
| `ROUNDTRIP_SUFFIX` | `8` | Held-out tail prefilled after load (exercises the load→prefill→generate path). |

Exit `0` / `PASS` = safe to set `QWEN_DISK_CACHE=1` and/or `QWEN_FRONT_SNAPSHOT=1`.
