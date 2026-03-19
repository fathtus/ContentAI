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

# Shared state
_run_queue: queue.Queue = queue.Queue()
_is_running = threading.Event()


def stream_process(topic: str, language: str, articles: int,
                   topic2: str, language2: str, articles2: int,
                   dry_run: bool):
    """Run main.py as subprocess and push stdout/stderr lines to the queue."""
    _is_running.set()
    _run_queue.queue.clear()

    cmd = [
        sys.executable, str(BASE_DIR / "main.py"),
        "--topic", topic,
        "--language", language,
        "--articles", str(articles),
    ]
    if topic2.strip():
        cmd += ["--topic2", topic2, "--language2", language2, "--articles2", str(articles2)]
    if dry_run:
        cmd.append("--dry-run")

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
            _run_queue.put({"type": "log", "data": line.rstrip()})
        proc.wait()
        if proc.returncode == 0:
            _run_queue.put({"type": "done", "data": "Pipeline completed successfully."})
        else:
            _run_queue.put({"type": "error", "data": f"Process exited with code {proc.returncode}"})
    except Exception as e:
        _run_queue.put({"type": "error", "data": str(e)})
    finally:
        _is_running.clear()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_pipeline():
    if _is_running.is_set():
        return jsonify({"error": "Pipeline is already running."}), 409

    data = request.get_json()
    topic     = data.get("topic", "technology artificial intelligence").strip() or "technology artificial intelligence"
    language  = data.get("language", "en").strip() or "en"
    articles  = int(data.get("articles", 3))
    topic2    = data.get("topic2", "").strip()
    language2 = data.get("language2", "en").strip() or "en"
    articles2 = int(data.get("articles2", 3))
    dry_run   = bool(data.get("dry_run", False))

    thread = threading.Thread(
        target=stream_process,
        args=(topic, language, articles, topic2, language2, articles2, dry_run),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    """SSE endpoint — pushes log lines to the browser in real time."""
    def generate():
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            try:
                msg = _run_queue.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"type\": \"ping\"}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/status")
def status():
    return jsonify({"running": _is_running.is_set()})


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
