from object.origami_object import OrigamiObject, Face, Point, LineType
import torch
import math
# Solver
# 1. Calculate face normals of all triangular faces in mesh (one face per thread).
# 2. Calculate current fold angle for all edges in mesh (one edge per thread).
# 3. Calculate coefficients of Equations 3–6 for all edges in mesh (one edge per
# thread).
# 4. Calculate forces and velocities for all nodes in mesh (one node per thread).
# 5. Calculate positions for all nodes in mesh (one node per thread).

EA = 20.0
POINT_MASS = 1.0 # Mass of point
DT = 0.3 # Delta time

K_FOLD = 0.7
K_FACET = 0.7

K_FACE = 0.2


DAMPING_RATIO = 0.1

def clear(objectOrigami: OrigamiObject) -> None:
    # Clear previous step forces and verlocities
    for point in objectOrigami.listPoints:
        point.clear()
    return

def addAxialConstraintsForce(objectOrigami: OrigamiObject) -> None:
    for line in objectOrigami.listLines:
        p1 = objectOrigami.listPoints[line.p1Index]
        p2 = objectOrigami.listPoints[line.p2Index]
        l0 = OrigamiObject.getOriginalDistance(p1,p2)
        force = EA/l0 * (OrigamiObject.getDistance(p1, p2) - OrigamiObject.getOriginalDistance(p1, p2))
        unitVector = OrigamiObject.getUnitVector(p1, p2)
        force = force * unitVector

        p1.force += force
        p2.force -= force
    return

def addCreaseConstraintsForce(objectOrigami: OrigamiObject) -> None:
    # TODO: Implement
    global bruh
   
    for lineIndex, faceIndices in objectOrigami.mappingLineToFace.items():
        if len(faceIndices)!=2: continue
        if len(faceIndices)==0 or len(faceIndices)>=3: raise ValueError(f"len(faceIndices) {len(faceIndices)} is not 1 or 2")
        
        K_CREASE = 0
        if (objectOrigami.listLines[lineIndex].lineType == LineType.MOUNTAIN or
            objectOrigami.listLines[lineIndex].lineType == LineType.VALLEY 
            ):
            l0 = objectOrigami.listLines[lineIndex].get_line_original_length(objectOrigami.listPoints)
            K_CREASE = l0*K_FOLD
        elif (objectOrigami.listLines[lineIndex].lineType==LineType.FACET):
            l0 = objectOrigami.listLines[lineIndex].get_line_original_length(objectOrigami.listPoints)
            K_CREASE = l0*K_FACET
     
        theta = objectOrigami.calculate_theta(lineIndex)
        target_theta = objectOrigami.listLines[lineIndex].targetTheta
        
        force = -K_CREASE*(theta-target_theta)
        p3_index =objectOrigami.listLines[lineIndex].p1Index
       
        p4_index =objectOrigami.listLines[lineIndex].p2Index

        face1 = objectOrigami.listFaces[objectOrigami.mappingLineToFace[lineIndex][0]]
        face2 = objectOrigami.listFaces[objectOrigami.mappingLineToFace[lineIndex][1]]

        p3 = objectOrigami.listPoints[p3_index]
        p4 = objectOrigami.listPoints[p4_index]
        p1, p2 = p3, p3 
        #p4 p2 p3
        #p2 p3 p4
        #p3 p4 p2
        # p3, p4, p1 

        if face1.point1Index == p3_index:
            if face1.point2Index == p4_index:
                p2 = objectOrigami.listPoints[face1.point3Index]
            else:
                p2 = objectOrigami.listPoints[face1.point2Index]
        elif face1.point2Index == p3_index:
            if face1.point1Index == p4_index:
                p2 = objectOrigami.listPoints[face1.point3Index]
            else:
                p2 = objectOrigami.listPoints[face1.point1Index]
        elif face1.point3Index == p3_index:
            if face1.point1Index == p4_index:
                p2 = objectOrigami.listPoints[face1.point2Index]
            else:
                p2 = objectOrigami.listPoints[face1.point1Index]

        #p3 p1 p4
        #p4 p3 p1
        #p1 p4 p3
        if face2.point1Index == p3_index:
            if face2.point2Index == p4_index:
                p1 = objectOrigami.listPoints[face2.point3Index]
            else:
                p1 = objectOrigami.listPoints[face2.point2Index]
        elif face2.point2Index == p3_index:
            if face2.point1Index == p4_index:
                p1 = objectOrigami.listPoints[face2.point3Index]
            else:
                p1 = objectOrigami.listPoints[face2.point1Index]
        elif face2.point3Index == p3_index:
            if face2.point1Index == p4_index:
                p1 = objectOrigami.listPoints[face2.point2Index]
            else:
                p1 = objectOrigami.listPoints[face2.point1Index]
      
        n1 = face2.calculate_and_update_normal(objectOrigami.listPoints)
        n2 = face1.calculate_and_update_normal(objectOrigami.listPoints)
        h1 = OrigamiObject.calculate_distance_point_to_line_2(p1,p3,p4)
        h2 = OrigamiObject.calculate_distance_point_to_line_2(p2,p3,p4)
        alpha_4_31, alpha_3_14, alpha_1_43 = OrigamiObject.calculate_face_angles(p4,p3,p1)
        alpha_4_23, alpha_3_42, alpha_2_43 = OrigamiObject.calculate_face_angles(p4,p3,p2)
        cot_4_31 = 1.0/torch.tan(alpha_4_31)
        cot_3_14 = 1.0/torch.tan(alpha_3_14)

        cot_4_23 = 1.0/torch.tan(alpha_4_23)
        cot_3_42 = 1.0/torch.tan(alpha_3_42)

        p1.force += force*n1/h1
        p2.force += force*n2/h2
       
        p3.force += force*(-cot_4_31/(cot_4_31+cot_3_14)*n1/h1-cot_4_23/(cot_4_23+cot_3_42)*n2/h2)
        p4.force += force*(-cot_3_14/(cot_4_31+cot_3_14)*n1/h1-cot_3_42/(cot_4_23+cot_3_42)*n2/h2)
   
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
        a1, a2, a3 = OrigamiObject.calculate_face_angles(p1, p2, p3)

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
        l0 = OrigamiObject.getOriginalDistance(p1,p2)
        force =  2*DAMPING_RATIO*math.sqrt(EA/l0*POINT_MASS) * (p2.verlocity - p1.verlocity)
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
    addAxialConstraintsForce(objectOrigami)  # co van de
    addCreaseConstraintsForce(objectOrigami) # Bi fail do facet gap qua nhieu
    addFaceConstraintsForce(objectOrigami)

    calculateVelocities(objectOrigami)
    addDampingForce(objectOrigami)
    calculateNewPositions(objectOrigami)
    return