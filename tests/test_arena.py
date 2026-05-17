import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import ArenaConfig
from src.arena import Environment, SafeZone, RAMNode


def test_safe_zone_contains():
    sz = SafeZone(center=(0, 0, 0), initial_radius=10, contract_rate=0.5, contract_interval=1)
    assert sz.contains((0, 0, 0))
    assert sz.contains((5, 5, 5))
    assert not sz.contains((20, 0, 0))


def test_safe_zone_contracts():
    sz = SafeZone(center=(0, 0, 0), initial_radius=10, contract_rate=0.5, contract_interval=1)
    sz.tick()
    assert sz.radius == 5.0
    sz.tick()
    assert sz.radius == 2.5


def test_safe_zone_contract_interval():
    sz = SafeZone(center=(0, 0, 0), initial_radius=10, contract_rate=0.5, contract_interval=5)
    for _ in range(4):
        sz.tick()
    assert sz.radius == 10.0
    sz.tick()
    assert sz.radius == 5.0


def test_ram_node_collect():
    node = RAMNode((0, 0, 0), yield_amount=15)
    assert node.collect() == 15
    assert node.collect() == 0


def test_ram_node_distance():
    node = RAMNode((0, 0, 0))
    assert node.distance_to((3, 4, 0)) == 5.0


def test_environment_tick():
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    env.tick()
    assert env.tick_count == 1


def test_environment_ram_spawn():
    config = ArenaConfig(
        grid_size=(100, 100, 100),
        ram_node_spawn_interval=5,
        max_ram_nodes=10,
    )
    env = Environment(config)
    for _ in range(5):
        env.tick()
    assert len(env.active_ram_nodes) >= 1


def test_random_position_in_safe_zone():
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    for _ in range(10):
        pos = env.random_position_in_safe_zone()
        assert env.is_in_safe_zone(pos), f"{pos} not in safe zone"
        assert env.is_in_bounds(pos), f"{pos} out of bounds"


def test_collect_nearest_ram():
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))
    env.ram_nodes.clear()
    env.ram_nodes.append(RAMNode((10, 10, 10), 15))
    env.ram_nodes.append(RAMNode((50, 50, 50), 15))

    collected = env.collect_nearest_ram((9, 9, 9), range=5.0)
    assert collected == 15, "Should collect nearest RAM node within range"

    collected2 = env.collect_nearest_ram((90, 90, 90), range=5.0)
    assert collected2 == 0, "Should not collect if out of range"


def test_resolve_collisions():
    env = Environment(ArenaConfig(grid_size=(100, 100, 100)))

    class FakeAgent:
        def __init__(self, pos):
            self.position = pos

    agents = [
        FakeAgent((10, 10, 10)),
        FakeAgent((10, 10, 10)),
        FakeAgent((20, 20, 20)),
    ]
    collisions = env.resolve_collisions(agents)
    assert len(collisions) >= 1


if __name__ == "__main__":
    test_safe_zone_contains()
    print("[PASS] test_safe_zone_contains")
    test_safe_zone_contracts()
    print("[PASS] test_safe_zone_contracts")
    test_safe_zone_contract_interval()
    print("[PASS] test_safe_zone_contract_interval")
    test_ram_node_collect()
    print("[PASS] test_ram_node_collect")
    test_ram_node_distance()
    print("[PASS] test_ram_node_distance")
    test_environment_tick()
    print("[PASS] test_environment_tick")
    test_environment_ram_spawn()
    print("[PASS] test_environment_ram_spawn")
    test_random_position_in_safe_zone()
    print("[PASS] test_random_position_in_safe_zone")
    test_collect_nearest_ram()
    print("[PASS] test_collect_nearest_ram")
    test_resolve_collisions()
    print("[PASS] test_resolve_collisions")
    print("All arena tests passed!")
