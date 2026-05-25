import math
import random
from src.config import ArenaConfig
from src.arena.safe_zone import SafeZone
from src.arena.ram_node import RAMNode


class Environment:
    def __init__(self, config: ArenaConfig):
        self.config = config
        self.tick_count = 0

        gx, gy, gz = config.grid_size
        self.grid_size = config.grid_size
        self.center = (gx / 2.0, gy / 2.0, gz / 2.0)

        self.safe_zone = SafeZone(
            center=self.center,
            initial_radius=config.safe_zone_initial_radius,
            contract_rate=config.safe_zone_contract_rate,
            contract_interval=config.safe_zone_contract_interval_ticks,
            contract_fixed=config.safe_zone_contract_fixed,
            min_radius=config.safe_zone_min_radius,
        )

        self.ram_nodes: list[RAMNode] = []

    def tick(self, agents=None):
        self.tick_count += 1
        alive_count = sum(1 for a in agents if a.alive) if agents else 0
        self.safe_zone.tick(alive_count)

        if self.tick_count % self.config.ram_node_spawn_interval == 0:
            num_nodes = random.randint(2, 3)
            for _ in range(num_nodes):
                self._spawn_ram_node(agents)

    def _spawn_ram_node(self, agents=None):
        if len(self.ram_nodes) >= self.config.max_ram_nodes:
            old = sorted(self.ram_nodes, key=lambda n: n.collected)[:10]
            for n in old:
                self.ram_nodes.remove(n)

        for _ in range(100):
            r = self.safe_zone.radius
            # 4. Safe Zone relevance: 90% near edge, 10% near center
            if random.random() < 0.9:
                dist = random.uniform(r * 0.8, r)
            else:
                dist = random.uniform(0, r * 0.2)
            
            theta = random.random() * math.pi * 2
            phi = math.acos(2 * random.random() - 1)
            x = self.center[0] + dist * math.sin(phi) * math.cos(theta)
            y = self.center[1] + dist * math.sin(phi) * math.sin(theta)
            z = self.center[2] + dist * math.cos(phi)
            pos = (
                max(0, min(self.grid_size[0] - 1, x)),
                max(0, min(self.grid_size[1] - 1, y)),
                max(0, min(self.grid_size[2] - 1, z)),
            )
            if agents:
                min_dist = min(
                    math.sqrt(sum((pos[i] - a.position[i]) ** 2 for i in range(3)))
                    for a in agents if a.alive
                ) if any(a.alive for a in agents) else float('inf')
                if min_dist < 3:
                    continue
            self.ram_nodes.append(RAMNode(pos, self.config.ram_node_yield))
            break

    def collect_nearest_ram(self, position: tuple, range: float = 2.0) -> int:
        nearest = None
        nearest_dist = range
        for node in self.ram_nodes:
            if node.collected:
                continue
            d = node.distance_to(position)
            if d < nearest_dist:
                nearest = node
                nearest_dist = d
        if nearest is not None:
            return nearest.collect()
        return 0

    def find_closest_ram(self, position: tuple) -> RAMNode:
        closest = None
        closest_dist = float("inf")
        for node in self.ram_nodes:
            if node.collected:
                continue
            d = node.distance_to(position)
            if d < closest_dist:
                closest = node
                closest_dist = d
        return closest

    def agent_positions(self, agents) -> list:
        return [a.position for a in agents]

    def resolve_collisions(self, agents) -> list:
        by_cell = {}
        for i, agent in enumerate(agents):
            key = self._grid_key(agent.position)
            by_cell.setdefault(key, []).append(i)

        collisions = []
        for key, indices in by_cell.items():
            if len(indices) > 1:
                collisions.append(indices)
        return collisions

    def _grid_key(self, pos: tuple) -> tuple:
        return (int(pos[0]), int(pos[1]), int(pos[2]))

    def is_in_bounds(self, pos: tuple) -> bool:
        x, y, z = pos
        gx, gy, gz = self.grid_size
        return 0 <= x < gx and 0 <= y < gy and 0 <= z < gz

    def is_in_safe_zone(self, pos: tuple) -> bool:
        return self.safe_zone.contains(pos)

    def safe_zone_damage(self, pos: tuple) -> float:
        return self.safe_zone.damage_at_point(pos, self.config.safe_zone_damage_rate, self.tick_count)

    def safe_zone_weight_corruption(self, pos: tuple) -> float:
        return self.safe_zone.weight_corruption_at_point(
            pos, self.config.safe_zone_weight_corruption_rate
        )

    def random_position_in_safe_zone(self, radius: float = None) -> tuple:
        r = (radius if radius is not None else self.safe_zone.radius) * (random.random() ** (1.0 / 3.0))
        theta = random.random() * math.pi * 2
        phi = math.acos(2 * random.random() - 1)
        x = self.center[0] + r * math.sin(phi) * math.cos(theta)
        y = self.center[1] + r * math.sin(phi) * math.sin(theta)
        z = self.center[2] + r * math.cos(phi)
        return (
            max(0, min(self.grid_size[0] - 1, x)),
            max(0, min(self.grid_size[1] - 1, y)),
            max(0, min(self.grid_size[2] - 1, z)),
        )

    @property
    def active_ram_nodes(self) -> list[RAMNode]:
        return [n for n in self.ram_nodes if not n.collected]

    def get_state_summary(self) -> dict:
        return {
            "tick": self.tick_count,
            "safe_zone_radius": self.safe_zone.radius,
            "safe_zone_volume_ratio": self.safe_zone.volume_ratio,
            "active_ram_nodes": len(self.active_ram_nodes),
            "total_ram_nodes_spawned": len(self.ram_nodes),
        }
