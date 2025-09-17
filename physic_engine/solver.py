from object.origami_object import OrigamiObject
import torch
import math
# Solver
# 1. Calculate face normals of all triangular faces in mesh (one face per thread).
# 2. Calculate current fold angle for all edges in mesh (one edge per thread).
# 3. Calculate coefficients of Equations 3–6 for all edges in mesh (one edge per
# thread).
# 4. Calculate forces and velocities for all nodes in mesh (one node per thread).
# 5. Calculate positions for all nodes in mesh (one node per thread).

K_AXIAL = 100.0
POINT_MASS = 1.0 # Mass of point
DT = 0.01 # Delta time
DAMPING_RATIO = 0.1
VISCOUS_DAMPING_COEFFICIENT = 2*DAMPING_RATIO*math.sqrt(K_AXIAL*POINT_MASS)

def clear(objectOrigami: OrigamiObject) -> None:
    # Clear previous step forces and verlocities
    for point in objectOrigami.listPoints:
        point.clear()
    return

def addAxialConstraintsForce(objectOrigami: OrigamiObject) -> None:
    for line in objectOrigami.listLines:
        p1 = objectOrigami.listPoints[line.p1Index]
        p2 = objectOrigami.listPoints[line.p2Index]
        force = K_AXIAL * (OrigamiObject.getDistance(p1, p2) - OrigamiObject.getOriginalDistance(p1, p2))
        unitVector = OrigamiObject.getUnitVector(p1, p2)
        force = force * unitVector
        p1.force += force
        p2.force -= force
    return

def addCreaseConstraintsForce(objectOrigami: OrigamiObject) -> None:
    return

def addFaceConstraintsForce(objectOrigami: OrigamiObject) -> None:
    return

def calculateVelocities(objectOrigami: OrigamiObject) -> None:
    for point in objectOrigami.listPoints:
        a = point.force / POINT_MASS
        point.verlocity += a * DT
    return

def addDampingForce(objectOrigami: OrigamiObject) -> None:
    for line in objectOrigami.listLines:
        p1 = objectOrigami.listPoints[line.p1Index]
        p2 = objectOrigami.listPoints[line.p2Index]
        force = VISCOUS_DAMPING_COEFFICIENT * (p2.verlocity - p1.verlocity)
        p1.force += force
        p2.force -= force
    return

def calculateNewPositions(objectOrigami: OrigamiObject) -> None:
    for point in objectOrigami.listPoints:
        a = point.force / POINT_MASS
        point.position += a * DT * DT
    return

def solverStep(objectOrigami: OrigamiObject) -> None:
    clear(objectOrigami)
    addAxialConstraintsForce(objectOrigami)
    addCreaseConstraintsForce(objectOrigami)
    addFaceConstraintsForce(objectOrigami)

    calculateVelocities(objectOrigami)
    addDampingForce(objectOrigami)
    calculateNewPositions(objectOrigami)
    return