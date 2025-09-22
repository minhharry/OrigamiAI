from visualization.show_origami_object import show_origami_object, show_origami_object_2d, show_origami_object_2d_new, show_faces_2d
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from utils.get_points_line_from_svg import get_points_line_from_svg
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from physic_engine.solver import solverStep
from visualization.animate import show_origami_object_open3d,show_origami_object_open3d_new
import torch
import sys

def main():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object(o)
    show_origami_object_2d(o,True)
    for face in o.listFaces:
        print(face)
    show_faces_2d(o)

def main2():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    print("points:", len(o.listPoints))
    print("lines:", len(o.listLines))
    print("faces:", len(o.listFaces))
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
    # listPoints[8].position += torch.tensor([0, 1, 0], dtype=torch.float32)
    # show_origami_object_open3d(o, solverStep)
    show_origami_object_open3d_new(o,solverStep,30,True,True,True,True,2)

def main4():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    p1 = Point(1,0,0)
    p2 = Point(0,1,0)
    p3 = Point(0,2,0)
    print("d: ",OrigamiObject.calculate_distance_point_to_line_2(p1,p2,p3))
    show_origami_object_2d_new(o,True,True)

if __name__ == "__main__":
    main3()
