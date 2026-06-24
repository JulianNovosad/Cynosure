"""Real-time web visualization dashboard for embodied learning battleground."""
import numpy as np
import threading
import json
import os
from typing import List, Tuple, Dict, Optional
from flask import Flask, render_template, jsonify
from datetime import datetime


class EmbodiedVisualizer:
    """
    Flask-based web visualizer for embodied agent learning.

    Tracks agent state, enemies, obstacles, and episode rewards.
    Serves a real-time web dashboard at http://localhost:PORT/
    """

    def __init__(self, port: int = 8080):
        """
        Initialize the visualizer.

        Args:
            port: Port number for Flask server (default 8080)
        """
        self.port = port

        # Determine absolute paths for templates and static files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(base_dir, 'templates')
        static_dir = os.path.join(base_dir, 'static')

        self.app = Flask(__name__,
                        template_folder=template_dir,
                        static_folder=static_dir)

        # Current state
        self._state = {
            'agent_pos': [0.0, 0.0],
            'goal_pos': [0.0, 0.0],
            'enemies': [],
            'obstacles': [],
            'episode': 0,
            'tick': 0,
            'current_reward': 0.0,
            'accumulated_reward': 0.0,
            'semantic_state': [0.0] * 64,
            'episode_rewards': [],  # List of [episode, reward] pairs
            'timestamp': datetime.now().isoformat(),
        }

        self._lock = threading.Lock()
        self._server_thread = None

        # Setup Flask routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            """Serve main dashboard page."""
            return render_template('index.html')

        @self.app.route('/api/state')
        def get_state():
            """Get current state as JSON."""
            with self._lock:
                return jsonify(self._state)

    def update_state(
        self,
        agent_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        enemies: List[Dict],
        obstacles: List[Tuple[float, float]],
        episode: int,
        tick: int,
        current_reward: float,
        accumulated_reward: float,
        semantic_state: np.ndarray,
    ):
        """
        Update the visualization state.

        Args:
            agent_pos: (x, y) tuple of agent position
            goal_pos: (x, y) tuple of goal position
            enemies: List of dicts with {position: [x, y], health: h, max_health: mh, strength: s}
            obstacles: List of (x, y) tuples representing obstacle positions
            episode: Current episode number
            tick: Current tick within episode
            current_reward: Reward from this tick
            accumulated_reward: Sum of all rewards in current episode
            semantic_state: 64D numpy array of semantic state
        """
        with self._lock:
            self._state['agent_pos'] = list(agent_pos)
            self._state['goal_pos'] = list(goal_pos)
            self._state['enemies'] = enemies
            self._state['obstacles'] = [[float(x), float(y)] for x, y in obstacles]
            self._state['episode'] = int(episode)
            self._state['tick'] = int(tick)
            self._state['current_reward'] = float(current_reward)
            self._state['accumulated_reward'] = float(accumulated_reward)
            self._state['semantic_state'] = semantic_state.astype(float).tolist()
            self._state['timestamp'] = datetime.now().isoformat()

    def log_episode_reward(self, episode: int, reward: float):
        """
        Log episode reward to history.

        Args:
            episode: Episode number
            reward: Total episode reward
        """
        with self._lock:
            self._state['episode_rewards'].append([int(episode), float(reward)])
            # Keep only last 100 episodes
            if len(self._state['episode_rewards']) > 100:
                self._state['episode_rewards'] = self._state['episode_rewards'][-100:]

    def start(self):
        """Start Flask server in background thread."""
        if self._server_thread is None or not self._server_thread.is_alive():
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self._server_thread.start()

    def _run_server(self):
        """Run Flask app in background (blocking call)."""
        try:
            self.app.run(
                host='localhost',
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            print(f"Error starting visualizer: {e}")

    def stop(self):
        """Stop the Flask server (if supported)."""
        # Note: Flask doesn't have a clean way to stop from another thread
        # in debug mode. This is a limitation of Flask's development server.
        # For production, use a proper WSGI server.
        pass
