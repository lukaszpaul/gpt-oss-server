"use strict";

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------
const state = {
  ws: null,
  mode: "ask",
  busy: false,
  file: null,        // currently open file path (relative)
  dirty: false,
  cur: null,         // { contentEl, thinkEl } for the in-flight assistant turn
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

// --------------------------------------------------------------------------
// Minimal markdown: escape -> fenced code -> inline code -> bold
// --------------------------------------------------------------------------
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderMarkdown(text) {
  const parts = text.split(/```/);
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      const body = parts[i].replace(/^[a-zA-Z0-9_-]*\n/, "");
      html += "<pre><code>" + esc(body) + "</code></pre>";
    } else {
      let chunk = esc(parts[i]);
      chunk = chunk.replace(/`([^`]+)`/g, "<code>$1</code>");
      chunk = chunk.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      html += chunk;
    }
  }
  return html;
}

// --------------------------------------------------------------------------
// Health / model status
// --------------------------------------------------------------------------
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const s = $("#model-status");
    s.textContent = "model: " + (h.model_up ? h.model_id : "down");
    s.className = "status " + (h.model_up ? "up" : "down");
    if (h.root) {
      $("#folder-input").value = h.root;
      $("#ws-name").textContent = h.root;
    }
  } catch (_) {}
}

// --------------------------------------------------------------------------
// File explorer
// --------------------------------------------------------------------------
async function openFolder() {
  const path = $("#folder-input").value.trim();
  if (!path) { alert("Type a folder path in the box first."); return; }
  let r, data;
  try {
    r = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    data = await r.json();
  } catch (err) {
    alert("Could not reach the backend.\n\nMake sure you opened this page at " +
          "http://127.0.0.1:8001 (served by ide_backend.py) — not the file " +
          "directly or a preview panel.\n\n(" + err + ")");
    return;
  }
  if (!r.ok) { alert(data.error || "Could not open folder"); return; }
  $("#ws-name").textContent = data.root;
  const tree = $("#tree");
  tree.innerHTML = "";
  renderEntries(tree, data.entries);
}

function renderEntries(container, entries) {
  for (const e of entries) {
    const li = el("li");
    const row = el("div", "row");
    row.dataset.path = e.path;
    row.dataset.type = e.type;
    const twisty = el("span", "twisty", e.type === "dir" ? "▸" : "");
    const icon = el("span", "icon", e.type === "dir" ? "📁" : "📄");
    const name = el("span", "name", e.name);
    row.append(twisty, icon, name);
    li.append(row);
    container.append(li);

    if (e.type === "dir") {
      const childUl = el("ul", "hidden");
      li.append(childUl);
      row.addEventListener("click", async () => {
        if (childUl.dataset.loaded !== "1") {
          const rr = await fetch("/api/dir?path=" + encodeURIComponent(e.path));
          const dd = await rr.json();
          if (rr.ok) { renderEntries(childUl, dd.entries); childUl.dataset.loaded = "1"; }
        }
        const open = childUl.classList.toggle("hidden");
        twisty.textContent = open ? "▸" : "▾";
      });
    } else {
      row.addEventListener("click", () => openFile(e.path, row));
    }
  }
}

async function openFile(path, row) {
  if (state.dirty && !confirm("Discard unsaved changes?")) return;
  const r = await fetch("/api/file?path=" + encodeURIComponent(path));
  const data = await r.json();
  if (!r.ok) { alert(data.error || "Could not open file"); return; }
  state.file = data.path;
  state.dirty = false;
  $("#editor").value = data.content;
  $("#editor-title").textContent = data.path;
  $("#editor-title").classList.remove("muted");
  $("#dirty-dot").classList.add("hidden");
  $("#save-btn").disabled = true;
  document.querySelectorAll("#tree .row.active").forEach((n) => n.classList.remove("active"));
  if (row) row.classList.add("active");
}

async function saveFile() {
  if (!state.file) return;
  const r = await fetch("/api/file", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: state.file, content: $("#editor").value }),
  });
  if (!r.ok) { const d = await r.json(); alert(d.error || "Save failed"); return; }
  state.dirty = false;
  $("#dirty-dot").classList.add("hidden");
  $("#save-btn").disabled = true;
}

// --------------------------------------------------------------------------
// Chat / WebSocket
// --------------------------------------------------------------------------
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onmessage = (ev) => handleEvent(JSON.parse(ev.data));
  ws.onclose = () => setTimeout(connectWS, 1500);
}

const messages = () => $("#messages");
function scrollDown() { messages().scrollTop = messages().scrollHeight; }

function addUserMsg(text) {
  const wrap = el("div", "msg user");
  wrap.append(el("div", "who", "you"));
  const bubble = el("div", "bubble");
  bubble.innerHTML = renderMarkdown(text);
  wrap.append(bubble);
  messages().append(wrap);
  scrollDown();
}

// Lazily create the assistant content bubble for the current turn segment.
function ensureContentEl() {
  if (state.cur && state.cur.contentEl) return state.cur.contentEl;
  const wrap = el("div", "msg assistant");
  wrap.append(el("div", "who", "assistant"));
  const bubble = el("div", "bubble");
  bubble._raw = "";
  wrap.append(bubble);
  messages().append(wrap);
  state.cur = state.cur || {};
  state.cur.contentEl = bubble;
  scrollDown();
  return bubble;
}

function ensureThinkEl() {
  if (state.cur && state.cur.thinkEl) return state.cur.thinkEl;
  const details = el("details", "think");
  details.open = false;
  const summary = el("summary", null, "💭 thinking");
  const bubble = el("div", "bubble");
  bubble._raw = "";
  details.append(summary, bubble);
  messages().append(details);
  state.cur = state.cur || {};
  state.cur.thinkEl = bubble;
  scrollDown();
  return bubble;
}

const toolCards = {};

function handleEvent(msg) {
  switch (msg.type) {
    case "reasoning": {
      const b = ensureThinkEl();
      b._raw += msg.text;
      b.textContent = b._raw;
      scrollDown();
      break;
    }
    case "content": {
      const b = ensureContentEl();
      b._raw += msg.text;
      b.innerHTML = renderMarkdown(b._raw);
      scrollDown();
      break;
    }
    case "tool_start": {
      // New tool => the next content/thinking belongs to a fresh segment.
      state.cur = null;
      const card = el("div", "tool");
      const head = el("div", "tool-head");
      const tname = el("span", "tname", msg.name);
      const targs = el("span", "targs", argSummary(msg.name, msg.args));
      const tstatus = el("span", "tstatus run", "…");
      head.append(tname, targs, tstatus);
      const body = el("div", "tool-body hidden");
      const pre = el("pre");
      body.append(pre);
      head.addEventListener("click", () => body.classList.toggle("hidden"));
      card.append(head, body);
      messages().append(card);
      toolCards[msg.id] = { card, pre, tstatus, body };
      scrollDown();
      break;
    }
    case "tool_result": {
      const c = toolCards[msg.id];
      if (c) {
        c.tstatus.textContent = msg.status;
        c.tstatus.className = "tstatus " + msg.status;
        c.pre.textContent = msg.result;
        if (msg.status !== "ok") c.body.classList.remove("hidden");
      }
      // A file may have changed on disk: refresh the open editor if it matches.
      maybeReloadOpenFile(msg);
      break;
    }
    case "permission_request":
      renderPermission(msg);
      break;
    case "error": {
      const wrap = el("div", "msg assistant");
      const bubble = el("div", "bubble");
      bubble.style.borderColor = "var(--err)";
      bubble.style.color = "var(--err)";
      bubble.textContent = "⚠ " + msg.message;
      wrap.append(bubble);
      messages().append(wrap);
      scrollDown();
      break;
    }
    case "reset_ok":
      messages().innerHTML = "";
      break;
    case "done":
      setBusy(false);
      state.cur = null;
      break;
  }
}

function argSummary(name, args) {
  if (!args) return "";
  if (name === "run_git") return args.args || "";
  if (args.path != null) return args.path;
  return JSON.stringify(args);
}

function renderPermission(msg) {
  const card = el("div", "perm");
  card.append(el("div", null, "Allow this git command?"));
  card.append(el("div", "cmd", "$ " + msg.command));
  const actions = el("div", "perm-actions");
  const approve = el("button", null, "Approve");
  const deny = el("button", "deny", "Deny");
  actions.append(approve, deny);
  card.append(actions);
  messages().append(card);
  scrollDown();

  const respond = (approved) => {
    state.ws.send(JSON.stringify({ type: "permission", id: msg.id, approved }));
    card.classList.add("resolved");
    actions.innerHTML = "";
    actions.append(el("span", "muted", approved ? "✓ approved" : "✗ denied"));
  };
  approve.addEventListener("click", () => respond(true));
  deny.addEventListener("click", () => respond(false));
}

function maybeReloadOpenFile(msg) {
  if (msg.name !== "write_file" || msg.status !== "ok" || !state.file) return;
  // Re-fetch the open file silently if it's the one that was written.
  fetch("/api/file?path=" + encodeURIComponent(state.file))
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (d && !state.dirty) $("#editor").value = d.content;
    })
    .catch(() => {});
}

function sendChat() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text || state.busy) return;
  if (!state.ws || state.ws.readyState !== 1) { alert("Not connected."); return; }
  addUserMsg(text);
  state.cur = null;
  state.ws.send(JSON.stringify({ type: "chat", mode: state.mode, message: text }));
  input.value = "";
  setBusy(true);
}

function setBusy(b) {
  state.busy = b;
  $("#send-btn").disabled = b;
  $("#cancel-btn").classList.toggle("hidden", !b);
}

// --------------------------------------------------------------------------
// Wire up
// --------------------------------------------------------------------------
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#mode-toggle button").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  $("#mode-hint").textContent =
    mode === "agent" ? "AGENT · read + write + git (approval)" : "ASK · read-only";
}

function init() {
  $("#open-btn").addEventListener("click", openFolder);
  $("#folder-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") openFolder();
  });

  $("#save-btn").addEventListener("click", saveFile);
  $("#editor").addEventListener("input", () => {
    if (!state.file) return;
    state.dirty = true;
    $("#dirty-dot").classList.remove("hidden");
    $("#save-btn").disabled = false;
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault();
      saveFile();
    }
  });

  document.querySelectorAll("#mode-toggle button").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode)));

  $("#send-btn").addEventListener("click", sendChat);
  $("#chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
  $("#cancel-btn").addEventListener("click", () => {
    if (state.ws) state.ws.send(JSON.stringify({ type: "cancel" }));
  });
  $("#reset-btn").addEventListener("click", () => {
    if (state.ws) state.ws.send(JSON.stringify({ type: "reset" }));
  });

  setMode("ask");
  connectWS();
  refreshHealth();
  setInterval(refreshHealth, 8000);
}

document.addEventListener("DOMContentLoaded", init);
