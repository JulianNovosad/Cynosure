import logging
import math
from src.agent import Agent
from src.arena import Environment

# Action Indices aligned with Agent.py
ACTION_ATTACK = 0
ACTION_HUNT = 1
ACTION_COLLECT = 2
ACTION_MOVE = 3
ACTION_WHISPER = 4
ACTION_PROBE = 5
ACTION_MANIPULATE = 6

# Setup Logger
conflict_logger = logging.getLogger("conflicts")
conflict_logger.setLevel(logging.DEBUG)
if not conflict_logger.handlers:
    fh = logging.FileHandler("conflicts.log")
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    conflict_logger.addHandler(fh)

def log_conflict_attempt(tick, attacker, target, distance, outcome):
    a_pos = tuple(round(v, 2) for v in attacker.position)
    t_pos = tuple(round(v, 2) for v in target.position) if target else (0, 0, 0)
    a_action = attacker.last_action
    d_action = target.last_action if target else -1
    t_id = target.id if target else -1
    
    msg = f"TICK={tick} ATTACKER={attacker.id} TARGET={t_id} ATTACKER_POS={a_pos} TARGET_POS={t_pos} DIST={round(distance, 2)} ACTION_A={a_action} ACTION_D={d_action} OUTCOME={outcome}"
    conflict_logger.debug(msg)

def resolve_conflict(
    attacker: Agent,
    defender: Agent,
    env: Environment,
    agents: list,
    distance: float = 0.0,
    tick: int = 0,
):
    if defender and not defender.alive:
        log_conflict_attempt(tick, attacker, defender, distance, "TARGET_DEAD")
        return

    a_action = attacker.last_action
    
    # Range Checks — attack is short-range (collision-based), hunt is moderate
    if a_action == ACTION_ATTACK and distance > 3.0:
        log_conflict_attempt(tick, attacker, defender, distance, "RANGE_FAIL")
        return
    if a_action == ACTION_HUNT and distance > 8.0:
        log_conflict_attempt(tick, attacker, defender, distance, "RANGE_FAIL")
        return

    d_action = defender.last_action if defender else -1
    pair_key = (a_action, d_action)

    outcomes = {
        (ACTION_ATTACK, ACTION_ATTACK): _conflict_attack_attack,
        (ACTION_ATTACK, ACTION_HUNT): _conflict_attack_hunt,
        (ACTION_ATTACK, ACTION_MOVE): _conflict_attack_move,
        (ACTION_HUNT, ACTION_HUNT): _conflict_hunt_hunt,
        (ACTION_HUNT, ACTION_ATTACK): _conflict_hunt_attack,
        (ACTION_HUNT, ACTION_MOVE): _conflict_hunt_move,
    }

    handler = outcomes.get(pair_key)
    if handler:
        handler(attacker, defender, env, agents)
        log_conflict_attempt(tick, attacker, defender, distance, "SUCCESS")
    else:
        # Default handling for other pairs if they collide
        if a_action == ACTION_ATTACK:
             _conflict_attack_generic(attacker, defender, env, agents)
             log_conflict_attempt(tick, attacker, defender, distance, "SUCCESS")
        elif a_action == ACTION_HUNT:
             _conflict_hunt_generic(attacker, defender, env, agents)
             log_conflict_attempt(tick, attacker, defender, distance, "SUCCESS")
        else:
             log_conflict_attempt(tick, attacker, defender, distance, "NO_HANDLER")

def _conflict_attack_attack(a: Agent, d: Agent, env, agents):
    if a.ram >= d.ram:
        winner, loser = a, d
    else:
        winner, loser = d, a
    winner.ram += loser.ram * 0.5
    loser.alive = False

def _conflict_attack_hunt(a: Agent, d: Agent, env, agents):
    # Attacker has advantage over hunter
    a.ram += d.ram * 0.4
    d.alive = False

def _conflict_attack_move(a: Agent, d: Agent, env, agents):
    a.ram += d.ram * 0.3
    d.alive = False

def _conflict_hunt_hunt(a: Agent, d: Agent, env, agents):
    if a.ram >= d.ram:
        winner, loser = a, d
    else:
        winner, loser = d, a
    winner.ram += loser.ram * 0.2
    loser.alive = False

def _conflict_hunt_attack(a: Agent, d: Agent, env, agents):
    # Hunter vs Attacker (Attacker usually wins or does damage)
    d.ram += a.ram * 0.2
    a.ram *= 0.8

def _conflict_hunt_move(a: Agent, d: Agent, env, agents):
    stolen = min(15, d.ram)
    a.ram += stolen
    d.ram -= stolen

def _conflict_attack_generic(a: Agent, d: Agent, env, agents):
    a.ram += d.ram * 0.5
    d.alive = False

def _conflict_hunt_generic(a: Agent, d: Agent, env, agents):
    stolen = min(10, d.ram)
    a.ram += stolen
    d.ram -= stolen
