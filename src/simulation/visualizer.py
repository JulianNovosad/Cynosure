import threading
import os
from flask import Flask, send_from_directory, jsonify, make_response
from waitress import serve

app = Flask(__name__, static_folder=None)
FRONTEND_DIR = None
LATEST_STATE = {"tick": 0, "agents": [], "safe_zone": {"radius": 128}, "ram_nodes": [], "metrics": {}, "announcer": ""}
STATE_LOCK = threading.Lock()


@app.route("/")
def index():
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
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        serve(app, host="0.0.0.0", port=8080, threads=4, channel_timeout=5)

    def stop(self):
        self._running = False

    def push_state(self, state: dict):
        global LATEST_STATE
        with STATE_LOCK:
            LATEST_STATE.update(state)
