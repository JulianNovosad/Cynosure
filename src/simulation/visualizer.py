import threading
import os
import secrets
from flask import Flask, send_from_directory, jsonify, make_response, request as flask_request
from waitress import create_server

_ALLOWED_ORIGINS = {"http://localhost:8080", "http://127.0.0.1:8080"}
_VIS_API_TOKEN = secrets.token_urlsafe(32)

app = Flask(__name__, static_folder=None)
FRONTEND_DIR = None
LATEST_STATE = {"tick": 0, "agents": [], "safe_zone": {"radius": 128}, "ram_nodes": [], "metrics": {}, "announcer": ""}
STATE_LOCK = threading.Lock()

BRAIN_STATE = {}
BRAIN_STATE_LOCK = threading.Lock()
BRAIN_GRAPH_STATE = {}
BRAIN_GRAPH_STATE_LOCK = threading.Lock()

SEMANTIC_STATE = {}  # agent_id -> 57D semantic state vector
SEMANTIC_STATE_LOCK = threading.Lock()

REWARD_TRACE = {}  # agent_id -> list of (tick, reward) tuples
REWARD_TRACE_LOCK = threading.Lock()


def _validate_origin():
    origin = flask_request.headers.get("Origin", "")
    if origin and origin not in _ALLOWED_ORIGINS:
        return False
    return True


def _require_api_token():
    token = flask_request.headers.get("X-API-Token", "")
    # For localhost development, token can be skipped
    if _validate_origin():
        return True
    return secrets.compare_digest(token, _VIS_API_TOKEN)


@app.after_request
def add_security_headers(resp):
    origin = flask_request.headers.get("Origin", "")
    if origin in _ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "X-API-Token"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return resp


@app.route("/", methods=["GET", "OPTIONS"])
def index():
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    if FRONTEND_DIR is None:
        return "Frontend directory not set", 500
    resp = make_response(send_from_directory(FRONTEND_DIR, "index.html"))
    resp.cache_control.no_cache = True
    resp.cache_control.must_revalidate = True
    return resp


@app.route("/state", methods=["GET", "OPTIONS"])
def get_state():
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    with STATE_LOCK:
        snapshot = dict(LATEST_STATE)
    return jsonify(snapshot)


@app.route("/brain/<int:agent_id>", methods=["GET", "OPTIONS"])
def get_brain(agent_id):
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    if not _require_api_token():
        return jsonify({"error": "unauthorized"}), 401
    mode = flask_request.args.get("mode", "blocks")
    if mode not in ("blocks", "graph"):
        return jsonify({"error": "invalid mode"}), 400
    with BRAIN_STATE_LOCK:
        data = dict(BRAIN_STATE.get(agent_id, {"agent_id": agent_id, "tick": 0, "offline": True}))
    if mode == "graph":
        with BRAIN_GRAPH_STATE_LOCK:
            graph = dict(BRAIN_GRAPH_STATE.get(agent_id, {"agent_id": agent_id, "tick": 0, "offline": True}))
        return jsonify(graph)
    return jsonify(data)


@app.route("/brain-view", methods=["GET", "OPTIONS"])
def brain_view():
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    if not _require_api_token():
        return "unauthorized", 401
    if FRONTEND_DIR is None:
        return "Frontend directory not set", 500
    resp = make_response(send_from_directory(FRONTEND_DIR, "brain_view.html"))
    resp.cache_control.no_cache = True
    resp.cache_control.must_revalidate = True
    return resp


@app.route("/semantic/<int:agent_id>", methods=["GET", "OPTIONS"])
def get_semantic_state(agent_id):
    """Get the 57D semantic state vector for an agent."""
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    if not _require_api_token():
        return jsonify({"error": "unauthorized"}), 401

    with SEMANTIC_STATE_LOCK:
        data = SEMANTIC_STATE.get(agent_id, {
            "agent_id": agent_id,
            "tick": 0,
            "semantic_state": [0.0] * 57,
            "domains": {
                "existence": [],
                "sensory": [],
                "cognitive": [],
                "strategic": [],
                "tactical": [],
                "social": [],
                "value": [],
                "meta": []
            }
        })
    return jsonify(data)


@app.route("/reward/<int:agent_id>", methods=["GET", "OPTIONS"])
def get_reward_trace(agent_id):
    """Get reward trace history for an agent."""
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    if not _require_api_token():
        return jsonify({"error": "unauthorized"}), 401

    with REWARD_TRACE_LOCK:
        trace = REWARD_TRACE.get(agent_id, [])
    return jsonify({
        "agent_id": agent_id,
        "reward_trace": trace[-500:]  # Last 500 ticks
    })


@app.route("/semantic-view", methods=["GET", "OPTIONS"])
def semantic_view():
    """Serve the semantic state visualization frontend."""
    if flask_request.method == "OPTIONS":
        return make_response("", 204)
    if not _validate_origin():
        return make_response("", 403)
    if not _require_api_token():
        return "unauthorized", 401
    if FRONTEND_DIR is None:
        return "Frontend directory not set", 500
    resp = make_response(send_from_directory(FRONTEND_DIR, "semantic_view.html"))
    resp.cache_control.no_cache = True
    resp.cache_control.must_revalidate = True
    return resp


class SimulationVisualizer:
    def __init__(self, frontend_dir: str):
        global FRONTEND_DIR
        FRONTEND_DIR = os.path.abspath(frontend_dir)
        self._thread = None
        self._server = None

    def start(self, host: str = "127.0.0.1"):
        self._server = create_server(app, host=host, port=8080, threads=4, channel_timeout=5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"\n{'='*80}")
        print(f"Visualizer API Token: {_VIS_API_TOKEN}")
        print(f"Access semantic view: http://localhost:8080/semantic-view?token={_VIS_API_TOKEN}")
        print(f"{'='*80}\n")

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

    def push_semantic_state(self, agent_id: int, tick: int, semantic_vector: list):
        """Push 57D semantic state vector for visualization."""
        global SEMANTIC_STATE
        domains = {
            "existence": semantic_vector[0:8],
            "sensory": semantic_vector[8:16],
            "cognitive": semantic_vector[16:24],
            "strategic": semantic_vector[24:32],
            "tactical": semantic_vector[32:40],
            "social": semantic_vector[40:48],
            "value": semantic_vector[48:50],
            "meta": semantic_vector[50:57]
        }
        with SEMANTIC_STATE_LOCK:
            SEMANTIC_STATE[agent_id] = {
                "agent_id": agent_id,
                "tick": tick,
                "semantic_state": semantic_vector,
                "domains": domains
            }

    def push_reward(self, agent_id: int, tick: int, reward: float):
        """Push reward signal for trace visualization."""
        global REWARD_TRACE
        with REWARD_TRACE_LOCK:
            if agent_id not in REWARD_TRACE:
                REWARD_TRACE[agent_id] = []
            REWARD_TRACE[agent_id].append({"tick": tick, "reward": reward})
