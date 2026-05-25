import threading
import os
from flask import Flask, send_from_directory, jsonify, make_response
from waitress import create_server

app = Flask(__name__, static_folder=None)
FRONTEND_DIR = None
LATEST_STATE = {"tick": 0, "agents": [], "safe_zone": {"radius": 128}, "ram_nodes": [], "metrics": {}, "announcer": ""}
STATE_LOCK = threading.Lock()


@app.route("/")
def index():
    if FRONTEND_DIR is None:
        return "Frontend directory not set", 500
    resp = make_response(send_from_directory(FRONTEND_DIR, "index.html"))
    resp.cache_control.no_cache = True
    resp.cache_control.must_revalidate = True
    return resp


@app.route("/state")
def get_state():
    with STATE_LOCK:
        snapshot = dict(LATEST_STATE)
    return jsonify(snapshot)


class SimulationVisualizer:
    def __init__(self, frontend_dir: str):
        global FRONTEND_DIR
        FRONTEND_DIR = os.path.abspath(frontend_dir)
        self._thread = None
        self._server = None

    def start(self):
        self._server = create_server(app, host="0.0.0.0", port=8080, threads=4, channel_timeout=5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        if self._server:
            try:
                self._server.run()
            except Exception:
                # Suppress errors during shutdown (common with Waitress on Windows)
                pass

    def stop(self):
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        if self._thread:
            # Short timeout to avoid hanging the main process
            self._thread.join(timeout=1)

    def push_state(self, state: dict):
        global LATEST_STATE
        with STATE_LOCK:
            LATEST_STATE.update(state)
