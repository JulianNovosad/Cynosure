/**
 * Real-time visualization dashboard for embodied learning battleground.
 * Fetches state from /api/state and renders canvas + metrics.
 */

// Canvas and context references
const worldCanvas = document.getElementById('worldCanvas');
const worldCtx = worldCanvas.getContext('2d');
const learningCanvas = document.getElementById('learningCanvas');
const learningCtx = learningCanvas.getContext('2d');

// State tracking
let currentState = null;
let agentTrail = [];  // Last 20 positions for trail rendering

// Configuration
const GRID_SIZE = 20;  // World grid size (meters)
const CANVAS_SIZE = 500;  // Canvas size (pixels)
const PIXELS_PER_UNIT = CANVAS_SIZE / GRID_SIZE;
const UPDATE_INTERVAL = 50;  // ms (20 FPS)
const TRAIL_LENGTH = 20;

/**
 * Fetch current state from API.
 */
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok) {
            console.error('Failed to fetch state:', response.status);
            return null;
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching state:', error);
        return null;
    }
}

/**
 * Update all visual elements.
 */
function updateVisualization(state) {
    if (!state) return;

    currentState = state;

    // Update world canvas
    renderWorld(state);

    // Update metrics
    updateMetrics(state);

    // Update learning curve
    renderLearningCurve(state);
}

/**
 * Render the world state on canvas.
 */
function renderWorld(state) {
    // Clear canvas
    worldCtx.fillStyle = '#0a0e27';
    worldCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Draw grid background
    drawGrid();

    // Draw obstacles
    if (state.obstacles && state.obstacles.length > 0) {
        drawObstacles(state.obstacles);
    }

    // Draw goal
    if (state.goal_pos) {
        drawGoal(state.goal_pos);
    }

    // Draw enemies
    if (state.enemies && state.enemies.length > 0) {
        drawEnemies(state.enemies);
    }

    // Update and draw agent trail
    if (state.agent_pos) {
        agentTrail.push([state.agent_pos[0], state.agent_pos[1]]);
        if (agentTrail.length > TRAIL_LENGTH) {
            agentTrail.shift();
        }
        drawTrail();
    }

    // Draw agent
    if (state.agent_pos) {
        drawAgent(state.agent_pos);
    }
}

/**
 * Draw grid background.
 */
function drawGrid() {
    worldCtx.strokeStyle = 'rgba(0, 255, 0, 0.1)';
    worldCtx.lineWidth = 1;

    const cellSize = PIXELS_PER_UNIT;
    for (let i = 0; i <= GRID_SIZE; i++) {
        const px = i * cellSize;
        // Vertical lines
        worldCtx.beginPath();
        worldCtx.moveTo(px, 0);
        worldCtx.lineTo(px, CANVAS_SIZE);
        worldCtx.stroke();

        // Horizontal lines
        worldCtx.beginPath();
        worldCtx.moveTo(0, px);
        worldCtx.lineTo(CANVAS_SIZE, px);
        worldCtx.stroke();
    }
}

/**
 * Draw obstacles as gray squares.
 */
function drawObstacles(obstacles) {
    worldCtx.fillStyle = '#808080';
    obstacles.forEach(obs => {
        const [x, y] = obs;
        const px = x * PIXELS_PER_UNIT;
        const py = y * PIXELS_PER_UNIT;
        const size = 0.5 * PIXELS_PER_UNIT;  // 0.5m obstacle
        worldCtx.fillRect(px - size / 2, py - size / 2, size, size);
    });
}

/**
 * Draw goal as gold star.
 */
function drawGoal(goalPos) {
    const [x, y] = goalPos;
    const px = x * PIXELS_PER_UNIT;
    const py = y * PIXELS_PER_UNIT;

    // Draw gold circle
    worldCtx.fillStyle = '#FFD700';
    worldCtx.beginPath();
    worldCtx.arc(px, py, 8, 0, 2 * Math.PI);
    worldCtx.fill();

    // Draw star pattern
    worldCtx.strokeStyle = '#FFA500';
    worldCtx.lineWidth = 2;
    drawStar(px, py, 5, 8, 4);
}

/**
 * Draw a star at position (cx, cy).
 */
function drawStar(cx, cy, spikes, outerRadius, innerRadius) {
    let rot = Math.PI / 2 * 3;
    let step = Math.PI / spikes;

    worldCtx.beginPath();
    worldCtx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        worldCtx.lineTo(cx + Math.cos(rot) * outerRadius, cy + Math.sin(rot) * outerRadius);
        rot += step;
        worldCtx.lineTo(cx + Math.cos(rot) * innerRadius, cy + Math.sin(rot) * innerRadius);
        rot += step;
    }
    worldCtx.lineTo(cx, cy - outerRadius);
    worldCtx.closePath();
    worldCtx.stroke();
}

/**
 * Draw enemies as circles proportional to health.
 */
function drawEnemies(enemies) {
    enemies.forEach(enemy => {
        const [x, y] = enemy.position;
        const px = x * PIXELS_PER_UNIT;
        const py = y * PIXELS_PER_UNIT;

        // Color by health percentage
        const healthPct = enemy.health / enemy.max_health;
        const hue = healthPct * 120;  // Green (healthy) to red (low health)
        worldCtx.fillStyle = `hsl(${hue}, 100%, 50%)`;

        // Size by strength
        const radius = 3 + (enemy.strength / 80) * 5;

        worldCtx.beginPath();
        worldCtx.arc(px, py, radius, 0, 2 * Math.PI);
        worldCtx.fill();

        // Draw health bar above enemy
        drawHealthBar(px, py - 15, enemy.health, enemy.max_health, 20, 3);
    });
}

/**
 * Draw health bar above entity.
 */
function drawHealthBar(x, y, health, maxHealth, width, height) {
    // Background
    worldCtx.fillStyle = '#333333';
    worldCtx.fillRect(x - width / 2, y, width, height);

    // Health
    const healthWidth = (health / maxHealth) * width;
    worldCtx.fillStyle = health > maxHealth * 0.5 ? '#00FF00' : '#FF0000';
    worldCtx.fillRect(x - width / 2, y, healthWidth, height);

    // Border
    worldCtx.strokeStyle = '#00FF00';
    worldCtx.lineWidth = 1;
    worldCtx.strokeRect(x - width / 2, y, width, height);
}

/**
 * Draw agent trail as light blue path.
 */
function drawTrail() {
    if (agentTrail.length < 2) return;

    worldCtx.strokeStyle = 'rgba(0, 150, 255, 0.3)';
    worldCtx.lineWidth = 2;
    worldCtx.beginPath();

    const [startX, startY] = agentTrail[0];
    worldCtx.moveTo(startX * PIXELS_PER_UNIT, startY * PIXELS_PER_UNIT);

    for (let i = 1; i < agentTrail.length; i++) {
        const [x, y] = agentTrail[i];
        worldCtx.lineTo(x * PIXELS_PER_UNIT, y * PIXELS_PER_UNIT);
    }

    worldCtx.stroke();
}

/**
 * Draw agent as blue circle with direction indicator.
 */
function drawAgent(agentPos) {
    const [x, y] = agentPos;
    const px = x * PIXELS_PER_UNIT;
    const py = y * PIXELS_PER_UNIT;
    const radius = 6;

    // Agent circle
    worldCtx.fillStyle = '#0099FF';
    worldCtx.beginPath();
    worldCtx.arc(px, py, radius, 0, 2 * Math.PI);
    worldCtx.fill();

    // Direction indicator (forward direction)
    worldCtx.strokeStyle = '#00FF00';
    worldCtx.lineWidth = 2;
    worldCtx.beginPath();
    worldCtx.moveTo(px, py);
    worldCtx.lineTo(px + radius * 1.5, py);
    worldCtx.stroke();

    // Border
    worldCtx.strokeStyle = '#00FF00';
    worldCtx.lineWidth = 2;
    worldCtx.beginPath();
    worldCtx.arc(px, py, radius, 0, 2 * Math.PI);
    worldCtx.stroke();
}

/**
 * Update metrics display.
 */
function updateMetrics(state) {
    document.getElementById('episodeValue').textContent = state.episode;
    document.getElementById('tickValue').textContent = state.tick;
    document.getElementById('currentRewardValue').textContent = state.current_reward.toFixed(2);
    document.getElementById('accumulatedRewardValue').textContent = state.accumulated_reward.toFixed(2);
    document.getElementById('enemiesCount').textContent = state.enemies.length;

    // Threat level from semantic_state[8] (threat assessment domain start)
    const threatLevel = state.semantic_state[8] || 0.0;
    document.getElementById('threatLevel').textContent = threatLevel.toFixed(2);

    // Update semantic state samples
    document.getElementById('nav').textContent = state.semantic_state[0].toFixed(2);
    document.getElementById('threat').textContent = state.semantic_state[8].toFixed(2);
    document.getElementById('energy').textContent = state.semantic_state[23].toFixed(2);
}

/**
 * Render learning curve.
 */
function renderLearningCurve(state) {
    const data = state.episode_rewards || [];

    if (data.length === 0) {
        // Draw empty chart
        learningCtx.fillStyle = '#0a0e27';
        learningCtx.fillRect(0, 0, learningCanvas.width, learningCanvas.height);
        learningCtx.strokeStyle = '#00ff00';
        learningCtx.strokeRect(0, 0, learningCanvas.width, learningCanvas.height);
        return;
    }

    // Find min and max rewards
    const rewards = data.map(d => d[1]);
    const minReward = Math.min(...rewards);
    const maxReward = Math.max(...rewards);
    const rewardRange = maxReward - minReward || 1;

    // Clear canvas
    learningCtx.fillStyle = '#0a0e27';
    learningCtx.fillRect(0, 0, learningCanvas.width, learningCanvas.height);

    // Draw axes
    learningCtx.strokeStyle = '#00ff00';
    learningCtx.lineWidth = 1;
    learningCtx.beginPath();
    learningCtx.moveTo(40, 10);
    learningCtx.lineTo(40, learningCanvas.height - 20);
    learningCtx.lineTo(learningCanvas.width - 10, learningCanvas.height - 20);
    learningCtx.stroke();

    // Draw plot
    const padding = { left: 40, right: 10, top: 10, bottom: 20 };
    const plotWidth = learningCanvas.width - padding.left - padding.right;
    const plotHeight = learningCanvas.height - padding.top - padding.bottom;

    learningCtx.strokeStyle = '#00ff00';
    learningCtx.lineWidth = 2;
    learningCtx.beginPath();

    data.forEach((point, index) => {
        const [episode, reward] = point;
        const x = padding.left + (index / (data.length - 1 || 1)) * plotWidth;
        const y = padding.top + (1 - (reward - minReward) / rewardRange) * plotHeight;

        if (index === 0) {
            learningCtx.moveTo(x, y);
        } else {
            learningCtx.lineTo(x, y);
        }
    });

    learningCtx.stroke();

    // Draw axes labels
    learningCtx.fillStyle = '#00ff00';
    learningCtx.font = '10px Courier New';
    learningCtx.textAlign = 'right';
    learningCtx.fillText('Episode', learningCanvas.width - 5, learningCanvas.height - 5);
    learningCtx.save();
    learningCtx.translate(10, learningCanvas.height / 2);
    learningCtx.rotate(-Math.PI / 2);
    learningCtx.textAlign = 'center';
    learningCtx.fillText('Reward', 0, 0);
    learningCtx.restore();
}

/**
 * Main update loop.
 */
async function updateLoop() {
    const state = await fetchState();
    if (state) {
        updateVisualization(state);
    }
}

/**
 * Initialize and start the visualization.
 */
function initialize() {
    console.log('Initializing embodied learning visualizer...');
    // Start update loop
    setInterval(updateLoop, UPDATE_INTERVAL);
    // Initial update
    updateLoop();
}

// Start when page loads
window.addEventListener('load', initialize);
