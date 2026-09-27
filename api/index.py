import os
import json
import uuid
import base64
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ if present - no-op on Vercel, where env vars are set directly

import firebase_admin
from firebase_admin import credentials
from google.cloud import firestore as gcloud_firestore
from flask import Flask, request, jsonify, Response
from groq import Groq
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Firebase service account key, provided as a base64-encoded JSON string.
# (Base64 because most hosts, including Vercel, want env vars as one plain
# line - a raw JSON blob with newlines/quotes is easy to mangle when pasting.)
FIREBASE_CREDENTIALS_B64 = os.environ.get("FIREBASE_CREDENTIALS_B64")
# Firestore database ID. Leave as "(default)" unless you created your database
# under a custom ID in the console (the console shows this as "Database <id>"
# at the top of the Firestore page) - then set this env var to match.
FIRESTORE_DATABASE_ID = os.environ.get("FIRESTORE_DATABASE_ID", "(default)")

# --- Efficiency knobs -------------------------------------------------------
# These four control token spend directly. Lower = cheaper/faster, at some
# cost to how much context the model has. Tune via env vars, no code changes.
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "8"))
MAX_HISTORY_CHARS = int(os.environ.get("MAX_HISTORY_CHARS", "4000"))  # hard cap regardless of message count
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "600"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))  # lower = more literal/less "creative" = fewer hallucinations

app = Flask(__name__)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Firebase / Firestore init ----------------------------------------------
firestore_db = None
if FIREBASE_CREDENTIALS_B64:
    try:
        cred_json = json.loads(base64.b64decode(FIREBASE_CREDENTIALS_B64))
        cred = credentials.Certificate(cred_json)
        if not firebase_admin._apps:  # avoid re-initializing on hot reload
            firebase_admin.initialize_app(cred)
        # Built directly from the google-cloud-firestore client (rather than
        # firebase_admin.firestore.client()) because this pinned version of
        # firebase-admin doesn't expose a way to pick a non-default database
        # ID, and the Google Cloud console lets you name your database
        # anything when you create it (shown as "Database <id>" at the top
        # of the Firestore page there).
        firestore_db = gcloud_firestore.Client(
            project=cred_json.get("project_id"),
            credentials=cred.get_credential(),
            database=FIRESTORE_DATABASE_ID,
        )
    except Exception as e:
        print(f"Firebase init failed: {e}")

# Kept short on purpose: every word here is sent on *every single request*.
# A bloated system prompt is one of the most common silent sources of wasted tokens.
SYSTEM_PROMPT = (
    "You are a direct, efficient personal AI agent. Use the `web_search` tool "
    "only for things you'd otherwise be guessing at (current events, releases, "
    "prices, specific facts) - never for things you already know confidently. "
    "Answer concisely: no filler, no repeating the question, no over-hedging. "
    "If you are not confident an answer is fully correct, say so briefly in the answer "
    "itself rather than stating it as fact.\n\n"
    "End every reply on its own final line with exactly: CONF: high | medium | low\n"
    "- high: you are certain / it's a well-established fact you verified or already knew solidly\n"
    "- medium: reasonably sure but there's some room for error or it's partly inference\n"
    "- low: you are guessing, extrapolating, or the web_search results were thin/unclear\n"
    "Never omit this line, and never explain it - just the tag."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information (news, releases, facts).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_messages_collection():
    return firestore_db.collection("messages")


def init_db():
    # Firestore has no schema/tables and creates collections on first write,
    # so there's nothing to set up ahead of time. Kept as a no-op so the rest
    # of the code (and the /api/chat route) doesn't need to change.
    return


def load_history(session_id):
    if not firestore_db:
        return []
    coll = get_messages_collection()
    # Filter by session_id only, then sort in Python - this avoids needing a
    # Firestore composite index (session_id + created_at), which Firestore
    # would otherwise ask you to create via a console link on first query.
    docs = coll.where("session_id", "==", session_id).stream()
    rows = [
        {"role": d.get("role"), "content": d.get("content"), "created_at": d.get("created_at")}
        for d in (doc.to_dict() for doc in docs)
    ]
    rows.sort(key=lambda m: m["created_at"])
    rows = rows[-MAX_HISTORY_MESSAGES:]
    rows = [{"role": m["role"], "content": m["content"]} for m in rows]
    return trim_history(rows)


def trim_history(history):
    """Keep the most recent messages, dropping older ones once we hit the
    character budget. Character count is a cheap stand-in for token count
    (roughly 4 chars/token) - good enough to keep requests small without
    calling a tokenizer."""
    kept = []
    total_chars = 0
    for msg in reversed(history):
        total_chars += len(msg["content"])
        if total_chars > MAX_HISTORY_CHARS:
            break
        kept.append(msg)
    return list(reversed(kept))


def save_message(session_id, role, content):
    if not firestore_db:
        return
    coll = get_messages_collection()
    coll.add(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def run_web_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"- {r.get('title')}: {r.get('body')} ({r.get('href')})")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(session_id, user_message):
    if not groq_client:
        return {
            "reply": "Server is missing GROQ_API_KEY. Set it in your environment variables.",
            "confidence": "unknown",
        }

    history = load_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    save_message(session_id, "user", user_message)

    # Let the model call the search tool if it wants to, then answer.
    for _ in range(3):  # cap tool-call rounds so it can't loop forever
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        choice = completion.choices[0].message

        if choice.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
                }
            )
            for tool_call in choice.tool_calls:
                if tool_call.function.name == "web_search":
                    args = json.loads(tool_call.function.arguments or "{}")
                    # cap what the search returns too - long dumps of search
                    # results are one of the biggest token sinks in agents
                    result = run_web_search(args.get("query", user_message))
                else:
                    result = "Unknown tool."
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            continue  # ask the model again with tool results included

        raw_reply = choice.content or ""
        reply, confidence = extract_confidence(raw_reply)
        # store the clean reply (without the tag) so history doesn't waste
        # tokens re-sending "CONF: high" back to the model every turn
        save_message(session_id, "assistant", reply)
        return {"reply": reply, "confidence": confidence}

    return {"reply": "Sorry, I couldn't finish that request.", "confidence": "low"}


def extract_confidence(text):
    """Pull the trailing 'CONF: high|medium|low' tag off a reply. Falls back
    to 'unknown' if the model forgot it, rather than guessing."""
    lines = text.strip().splitlines()
    if not lines:
        return text, "unknown"
    last = lines[-1].strip().lower()
    for level in ("high", "medium", "low"):
        if last.startswith("conf") and level in last:
            body = "\n".join(lines[:-1]).strip()
            return body, level
    return text.strip(), "unknown"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        init_db()
    except Exception as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    result = run_agent(session_id, user_message)
    return jsonify(
        {
            "reply": result["reply"],
            "confidence": result["confidence"],
            "session_id": session_id,
        }
    )


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "groq_configured": bool(GROQ_API_KEY),
            "db_configured": bool(firestore_db),
        }
    )


# ---------------------------------------------------------------------------
# Frontend (single-file, no build step needed)
# ---------------------------------------------------------------------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent</title>
<style>
  :root {
    --bg: #0e0f10;
    --panel: #17181a;
    --border: #26282b;
    --text: #e8e6e1;
    --muted: #8a8d91;
    --accent: #5ee6a8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 18px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 10px;
  }
  header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    display: inline-block;
  }
  header h1 {
    font-size: 15px;
    font-weight: 500;
    margin: 0;
    letter-spacing: 0.3px;
  }
  header span.sub {
    color: var(--muted);
    font-size: 12px;
  }
  #log {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    max-width: 720px;
    width: 100%;
    margin: 0 auto;
  }
  .msg {
    margin-bottom: 22px;
    line-height: 1.55;
    font-size: 14px;
    white-space: pre-wrap;
  }
  .msg .who {
    color: var(--muted);
    font-size: 11px;
    text-transform: lowercase;
    margin-bottom: 4px;
  }
  .msg.user .who { color: var(--accent); }
  form {
    border-top: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    justify-content: center;
  }
  form .inner {
    max-width: 720px;
    width: 100%;
    display: flex;
    gap: 10px;
  }
  textarea {
    flex: 1;
    resize: none;
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    padding: 10px 12px;
    font-family: inherit;
    font-size: 14px;
    height: 46px;
  }
  textarea:focus { outline: 1px solid var(--accent); }
  button {
    background: var(--accent);
    color: #06120c;
    border: none;
    border-radius: 8px;
    padding: 0 18px;
    font-family: inherit;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
  .pending { color: var(--muted); font-style: italic; }
  .conf {
    display: inline-block;
    margin-top: 8px;
    font-size: 10px;
    letter-spacing: 0.3px;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .conf.high { color: #5ee6a8; border-color: #2c5e46; }
  .conf.medium { color: #e6c95e; border-color: #5e5330; }
  .conf.low { color: #e67a5e; border-color: #5e352c; }
  .conf.unknown { color: var(--muted); }
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>agent</h1>
    <span class="sub">groq + search + memory</span>
  </header>
  <div id="log"></div>
  <form id="form">
    <div class="inner">
      <textarea id="input" placeholder="Ask anything..." rows="1"></textarea>
      <button id="send" type="submit">Send</button>
    </div>
  </form>

<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let sessionId = localStorage.getItem('agent_session_id') || null;

function addMessage(who, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.innerHTML = '<div class="who">' + who + '</div>' + text.replace(/</g,'&lt;');
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;
  addMessage('user', text);
  const pending = addMessage('agent', '...');
  pending.classList.add('pending');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId })
    });
    const data = await res.json();
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem('agent_session_id', sessionId);
    }
    pending.classList.remove('pending');
    const conf = data.confidence || 'unknown';
    const confHtml = data.reply ? '<span class="conf ' + conf + '">confidence: ' + conf + '</span>' : '';
    pending.innerHTML = '<div class="who">agent</div>' + (data.reply || data.error || 'Error').replace(/</g,'&lt;') + '<br>' + confHtml;
  } catch (err) {
    pending.classList.remove('pending');
    pending.innerHTML = '<div class="who">agent</div>Request failed: ' + err;
  } finally {
    sendBtn.disabled = false;
    log.scrollTop = log.scrollHeight;
  }
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
</script>
</body>
</html>
"""

# Local dev entrypoint
if __name__ == "__main__":
    app.run(debug=True, port=5000)
