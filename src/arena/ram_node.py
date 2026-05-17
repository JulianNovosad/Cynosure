import math


class RAMNode:
    def __init__(self, position: tuple, yield_amount: int = 15):
        self.position = position
        self.yield_amount = yield_amount
        self.collected = False

    def distance_to(self, point: tuple) -> float:
        dx = self.position[0] - point[0]
        dy = self.position[1] - point[1]
        dz = self.position[2] - point[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def collect(self) -> int:
        if self.collected:
            return 0
        self.collected = True
        return self.yield_amount
