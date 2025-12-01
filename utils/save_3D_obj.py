from object.origami_object import Point, Line, Face
import os
def save_obj(points: list[Point], lines: list[Line], faces: list[Face], file_name: str, folder_path: str = "") -> None:
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