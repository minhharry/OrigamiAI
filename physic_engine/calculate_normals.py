import torch
from object.origami_object import OrigamiObject

def calculate_and_update_normals(o: OrigamiObject) -> list[torch.Tensor]:
    for face in o.listFaces:
        p1 = o.listPoints[face.point1Index].position
        p2 = o.listPoints[face.point2Index].position
        p3 = o.listPoints[face.point3Index].position
        v1 = p2 - p1
        v2 = p3 - p1
        normal = torch.cross(v1, v2)
        normal = normal / torch.norm(normal)
        face.normal = normal

    
if __name__ == "__main__":
    print("This is calculate_normals.py")
    calculate_and_update_normals()