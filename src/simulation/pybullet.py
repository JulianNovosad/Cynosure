# Mock pybullet for testing without the full library
import numpy as np

# Constants
GUI = 1
DIRECT = 0
WORLD_FRAME = 1
GEOM_PLANE = 2
GEOM_CYLINDER = 3
GEOM_BOX = 4
GEOM_SPHERE = 5

class MockPhysicsClient:
    def __init__(self, mode):
        self.mode = mode
        self.bodies = {}
        self.body_counter = 0
        self.gravity = (0, 0, -9.81)

    def connect(self, mode):
        return 0

    def disconnect(self, client_id):
        pass

_default_client = MockPhysicsClient(DIRECT)

def connect(mode):
    return 0

def disconnect(client_id):
    pass

def setAdditionalSearchPath(path):
    pass

def setGravity(x, y, z, physicsClientId=None):
    pass

def createCollisionShape(shape_type, radius=None, halfExtents=None, height=None, physicsClientId=None):
    return 1

def createMultiBody(baseMass, baseCollisionShapeIndex, basePosition, physicsClientId=None):
    return 1

def changeDynamics(body_id, link_id, lateralFriction=None, physicsClientId=None):
    pass

def getBasePositionAndOrientation(body_id, physicsClientId=None):
    return (0, 0, 0), (0, 0, 0, 1)

def getBaseVelocity(body_id, physicsClientId=None):
    return (0, 0, 0), (0, 0, 0)

def getContactPoints(bodyA=None, physicsClientId=None):
    return []

def applyExternalForce(body_id, link_id, force, position, frame, physicsClientId=None):
    pass

def stepSimulation(physicsClientId=None):
    pass

def rayTestClosest(from_pos, to_pos, physicsClientId=None):
    return -1, None, None, (0, 0, 0)
