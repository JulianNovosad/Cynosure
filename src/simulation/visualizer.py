import threading
import os
from flask import Flask, send_from_directory, jsonify, make_response
from waitress import create_server

app = Flask(__name__, static_folder=None)
FRONTEND_DIR = None
LATEST_STATE = {"tick": 0, "agents": [], "safe_zone": {"radius": 128}, "ram_nodes": [], "metrics": {}, "announcer": ""}
STATE_LOCK = threading.Lock()

BRAIN_STATE = {}
BRAIN_STATE_LOCK = threading.Lock()
BRAIN_GRAPH_STATE = {}
BRAIN_GRAPH_STATE_LOCK = threading.Lock()


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


@app.route("/brain/<int:agent_id>")
def get_brain(agent_id):
    from flask import request
    mode = request.args.get("mode", "blocks")
    with BRAIN_STATE_LOCK:
        data = dict(BRAIN_STATE.get(agent_id, {"agent_id": agent_id, "tick": 0, "offline": True}))
    if mode == "graph":
        with BRAIN_GRAPH_STATE_LOCK:
            graph = dict(BRAIN_GRAPH_STATE.get(agent_id, {"agent_id": agent_id, "tick": 0, "offline": True}))
        return jsonify(graph)
    return jsonify(data)


@app.route("/brain-view")
def brain_view():
    if FRONTEND_DIR is None:
        return "Frontend directory not set", 500
    resp = make_response(send_from_directory(FRONTEND_DIR, "brain_view.html"))
    resp.cache_control.no_cache = True
    resp.cache_control.must_revalidate = True
    return resp


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
                pass

    def stop(self):
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1)

    def push_state(self, state: dict):
        global LATEST_STATE
        with STATE_LOCK:
            LATEST_STATE.update(state)

    def push_brain_state(self, agent_id: int, data: dict):
        global BRAIN_STATE
        with BRAIN_STATE_LOCK:
            BRAIN_STATE[agent_id] = data

    def push_brain_graph_state(self, agent_id: int, data: dict):
        global BRAIN_GRAPH_STATE
        with BRAIN_GRAPH_STATE_LOCK:
            BRAIN_GRAPH_STATE[agent_id] = data
