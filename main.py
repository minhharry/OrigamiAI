from visualization.show_origami_object import show_origami_object, show_origami_object_2d, show_origami_object_2d_new, show_faces_2d
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from utils.get_points_line_from_svg import get_points_line_from_svg
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from physic_engine.solver import solverStep
from visualization.animate import show_origami_object_open3d
import torch

def main():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object(o)
    show_origami_object_2d(o)
    show_faces_2d(o)

def main2():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    print("points:", len(o.listPoints))
    print("lines:", len(o.listLines))
    print("faces:", len(o.listFaces))
    listPoints[8].position += torch.tensor([0, 200, 0], dtype=torch.float32)
    num_iterations = 100
    show_origami_object(o)
    for i in range(num_iterations):
        solverStep(o)
    show_origami_object(o)
    for i in range(num_iterations):
        solverStep(o)
    show_origami_object(o)

def main3():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    listPoints[8].position += torch.tensor([0, 500, 0], dtype=torch.float32)
    show_origami_object_open3d(o, solverStep)

if __name__ == "__main__":
    main3()
