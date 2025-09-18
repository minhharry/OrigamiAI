from object.origami_object import OrigamiObject, Face, Point
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

K_FACE = 10000.0


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

bruh = 0
def addCreaseConstraintsForce(objectOrigami: OrigamiObject) -> None:
    # TODO: Implement
    global bruh

    mappingLineToFace = {}
    for face_index in range(len(objectOrigami.listFaces)):
        face = objectOrigami.listFaces[face_index]
        if mappingLineToFace.get(face.line12Index) is None:
            mappingLineToFace[face.line12Index] = []
        if mappingLineToFace.get(face.line23Index) is None:
            mappingLineToFace[face.line23Index] = []
        if mappingLineToFace.get(face.line13Index) is None:
            mappingLineToFace[face.line13Index] = []
        mappingLineToFace[face.line12Index].append(face_index)
        mappingLineToFace[face.line23Index].append(face_index)
        mappingLineToFace[face.line13Index].append(face_index)

    if bruh == 0:
        print(mappingLineToFace)
        bruh = 1
    return

def addFaceConstraintsForce(objectOrigami: OrigamiObject) -> None:
    def calculateUnitVectorForce(face: Face, p1: Point, p2: Point, p3: Point) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate at point p2
        delta_p1 = torch.linalg.cross(face.normal, (p1.position - p2.position)) / torch.linalg.norm(p1.position - p2.position)
        delta_p3 = torch.linalg.cross(face.normal, (p2.position - p3.position)) / torch.linalg.norm(p2.position - p3.position)
        delta_p2 = - delta_p1 - delta_p3 
        return delta_p1, delta_p2, delta_p3
    
    for face in objectOrigami.listFaces:
        p1 = objectOrigami.listPoints[face.point1Index]
        p2 = objectOrigami.listPoints[face.point2Index]
        p3 = objectOrigami.listPoints[face.point3Index]

        face.calculate_and_update_normal(objectOrigami.listPoints)
        a1, a2, a3 = face.calculate_face_angles(p1, p2, p3)

        # Calculate unit vector force at point p2
        delta_p1, delta_p2, delta_p3 = calculateUnitVectorForce(face, p1, p2, p3)
        p1.force += K_FACE * (face.alpha1 - a1) * delta_p1
        p2.force += K_FACE * (face.alpha2 - a2) * delta_p2
        p3.force += K_FACE * (face.alpha3 - a3) * delta_p3

        # Calculate unit vector force at point p3
        delta_p1, delta_p2, delta_p3 = calculateUnitVectorForce(face, p2, p3, p1)
        p1.force += K_FACE * (face.alpha1 - a1) * delta_p1
        p2.force += K_FACE * (face.alpha2 - a2) * delta_p2        
        p3.force += K_FACE * (face.alpha3 - a3) * delta_p3

        # Calculate unit vector force at point p1
        delta_p1, delta_p2, delta_p3 = calculateUnitVectorForce(face, p3, p1, p2)
        p1.force += K_FACE * (face.alpha1 - a1) * delta_p1
        p2.force += K_FACE * (face.alpha2 - a2) * delta_p2
        p3.force += K_FACE * (face.alpha3 - a3) * delta_p3
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
    # addCreaseConstraintsForce(objectOrigami)
    addFaceConstraintsForce(objectOrigami)

    calculateVelocities(objectOrigami)
    addDampingForce(objectOrigami)
    calculateNewPositions(objectOrigami)
    return