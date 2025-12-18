from object.origami_object import Point, Line, Face
import os
import torch
def save_obj(points: list, lines: list, faces: list, file_name: str, folder_path: str = "") -> None:
    """
    Save 3D origami object to .obj
    points: list of Point
    lines: list of Line
    faces: list of faces (list of vertex indices)
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if folder_path == "":
        folder_path = os.path.join(BASE_DIR, "outputObj")

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w') as f:
        f.write("# Origami OBJ file\n")
        
        for p in points:
            x, y, z = float(p.position[0]), float(p.position[1]), float(p.position[2])
            f.write(f"v {x} {y} {z}\n")
        
        for face in faces:
            i1 = face.point1Index + 1
            i2 = face.point2Index + 1
            i3 = face.point3Index + 1
            f.write(f"f {i1} {i2} {i3}\n")

def save_obj_from_arrays(points, faces, file_name, folder_name=""):
    # print("save_obj_from_arrays")
    os.makedirs(folder_name, exist_ok=True)

    list_faces = []
    triangles = set()  # store unique triangles
    # print(points)
    # print(faces)
    with open(os.path.join(folder_name, file_name), "w") as f:
        # write vertices
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        # triangulate quads
        for face in faces:
            # triangle A: (0, 2, 3)
            
            tri1 = tuple(sorted([
                int(face[0]),
                int(face[2]),
                int(face[3])
            ]))

            # triangle B: (1, 2, 3)
            tri2 = tuple(sorted([
                int(face[1]),
                int(face[3]),
                int(face[2])
            ]))
            if tri1 not in triangles:
                list_faces.append([
                    int(face[0]),
                    int(face[2]),
                    int(face[3])
                ])
            if tri2 not in triangles:
                list_faces.append([
                    int(face[1]),
                    int(face[3]),
                    int(face[2])
                ])

            triangles.add(tri1)
            triangles.add(tri2)

        # write unique triangles (OBJ is 1-based)
        for tri in list_faces:
            f.write(
                f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n"
            )
