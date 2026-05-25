import math


class SafeZone:
    def __init__(
        self, center: tuple, initial_radius: float,
        contract_rate: float, contract_interval: int,
        contract_fixed: float = 0.0, min_radius: float = 5.0,
    ):
        self.center = center
        self.radius = initial_radius
        self.initial_radius = initial_radius
        self.contract_rate = contract_rate
        self.contract_interval = contract_interval
        self.contract_fixed = contract_fixed
        self.min_radius = min_radius
        self.ticks_since_contract = 0
        self._pause_counter = 0
        self._is_paused = False

    def tick(self, alive_count: int = None):
        if alive_count is not None and alive_count <= 2:
            self._pause_counter += 1
            self._is_paused = True
            if self._pause_counter >= 100:
                self._pause_counter = 0
                self._is_paused = False
        else:
            self._pause_counter = 0
            self._is_paused = False

        if self._is_paused:
            return

        self.ticks_since_contract += 1
        if self.ticks_since_contract >= self.contract_interval:
            if self.contract_fixed > 0:
                self.radius = max(self.min_radius, self.radius - self.contract_fixed)
            else:
                self.radius *= self.contract_rate
            self.ticks_since_contract = 0

    def contains(self, point: tuple) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        return (dx * dx + dy * dy + dz * dz) <= (self.radius * self.radius)

    def distance_to_boundary(self, point: tuple) -> float:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        dist_from_center = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dist_from_center - self.radius

    def distance_from_center(self, point: tuple) -> float:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def damage_at_point(self, point: tuple, damage_rate: float, tick: int = 0) -> float:
        outside = self.distance_to_boundary(point)
        if outside <= 0:
            return 0.0
        # Storm intensity increases over time
        intensity = 1.0 + (tick / 500.0)
        # Exponential damage based on distance + base penalty
        return (outside * damage_rate * 50.0 * intensity) + (5.0 * intensity)

    def weight_corruption_at_point(self, point: tuple, corruption_rate: float) -> float:
        outside = self.distance_to_boundary(point)
        if outside <= 0:
            return 0.0
        return outside * corruption_rate

    @property
    def volume_ratio(self) -> float:
        return (self.radius / self.initial_radius) ** 3
