"""
ContentAI Web Server
====================
Flask web interface to run the news-to-social pipeline and view results.
"""

import os
import sys
import json
import queue
import threading
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

# Independent state per pipeline
_queue1: queue.Queue = queue.Queue()
_running1 = threading.Event()

_queue2: queue.Queue = queue.Queue()
_running2 = threading.Event()


def _stream_process(q: queue.Queue, event: threading.Event, cmd: list):
    """Run main.py as subprocess and push stdout/stderr lines to the queue."""
    event.set()
    q.queue.clear()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR),
        )
        for line in proc.stdout:
            q.put({"type": "log", "data": line.rstrip()})
        proc.wait()
        if proc.returncode == 0:
            q.put({"type": "done", "data": "Pipeline completed successfully."})
        else:
            q.put({"type": "error", "data": f"Process exited with code {proc.returncode}"})
    except Exception as e:
        q.put({"type": "error", "data": str(e)})
    finally:
        event.clear()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run/page1", methods=["POST"])
def run_page1():
    if _running1.is_set():
        return jsonify({"error": "Page 1 pipeline is already running."}), 409

    data = request.get_json()
    topic     = data.get("topic", "technology artificial intelligence").strip() or "technology artificial intelligence"
    language  = data.get("language", "en").strip() or "en"
    articles  = int(data.get("articles", 3))
    dry_run   = bool(data.get("dry_run", False))
    platforms = data.get("platforms", ["facebook", "x", "instagram", "linkedin"])
    rewriter  = data.get("rewriter", "gemini")
    if not platforms:
        platforms = ["facebook", "x", "instagram", "linkedin"]

    cmd = [
        sys.executable, str(BASE_DIR / "main.py"),
        "--topic", topic,
        "--language", language,
        "--articles", str(articles),
        "--platforms", ",".join(platforms),
        "--rewriter", rewriter,
    ]
    if dry_run:
        cmd.append("--dry-run")

    threading.Thread(target=_stream_process, args=(_queue1, _running1, cmd), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/run/page2", methods=["POST"])
def run_page2():
    if _running2.is_set():
        return jsonify({"error": "Page 2 pipeline is already running."}), 409

    data = request.get_json()
    topic2    = data.get("topic2", "").strip()
    language2 = data.get("language2", "en").strip() or "en"
    articles2 = int(data.get("articles2", 3))
    dry_run   = bool(data.get("dry_run", False))
    rewriter  = data.get("rewriter", "gemini")

    if not topic2:
        return jsonify({"error": "Page 2 topic is required."}), 400

    cmd = [
        sys.executable, str(BASE_DIR / "main.py"),
        "--topic", "placeholder",   # required arg but skipped via --skip-page1
        "--topic2", topic2,
        "--language2", language2,
        "--articles2", str(articles2),
        "--skip-page1",
        "--rewriter", rewriter,
    ]
    if dry_run:
        cmd.append("--dry-run")

    threading.Thread(target=_stream_process, args=(_queue2, _running2, cmd), daemon=True).start()
    return jsonify({"status": "started"})


def _sse_generator(q: queue.Queue):
    yield "data: {\"type\": \"connected\"}\n\n"
    while True:
        try:
            msg = q.get(timeout=30)
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break
        except queue.Empty:
            yield "data: {\"type\": \"ping\"}\n\n"


@app.route("/stream/page1")
def stream_page1():
    return Response(_sse_generator(_queue1), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/stream/page2")
def stream_page2():
    return Response(_sse_generator(_queue2), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/status")
def status():
    return jsonify({"running1": _running1.is_set(), "running2": _running2.is_set()})


@app.route("/results")
def results():
    def read(p): return p.read_text() if p.exists() else None
    return jsonify({
        "draft":    read(BASE_DIR / "content_draft.md"),
        "report":   read(BASE_DIR / "publishing_report.md"),
        "draft2":   read(BASE_DIR / "content_draft_p2.md"),
        "report2":  read(BASE_DIR / "publishing_report_p2.md"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\nContentAI Web Interface running at http://localhost:{port}\n")
    app.run(debug=True, port=port, threaded=True)
