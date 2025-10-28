from visualization.show_origami_object import show_origami_object, show_origami_object_2d, show_origami_object_2d_new, show_faces_2d
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from utils.get_points_line_from_svg import get_points_line_from_svg, triangulate_all
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from physic_engine.solver import solverStep, setDeltaTime
from physic_engine.solver2 import OrigamiObjectMatrix, convert_to_matrix
from visualization.animate_pointcloud import Plotter
from visualization.animate import show_origami_object_open3d
from ptu.ptu import gen_ptu
import numpy as np
import time
import torch

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

def main_ptu():
    listPoints, listLines = gen_ptu(np.pi-0.01,9,5) # list[Point], list[Line], Line: {p1: Point, p2: Point, targetTheta: float}
    listLines_ = []
    # for i in range(len(listLines)):
    #     print(listLines[i].p1,listLines[i].p2)

    for i in range(len(listLines)):
        p1_index = listPoints.index(listLines[i].p1)
        p2_index = listPoints.index(listLines[i].p2)
        targetTheta = torch.tensor(listLines[i].targetTheta) if listLines[i].targetTheta != -999 else torch.tensor(0.0)
        lineType = LineType.VALLEY if targetTheta > 0 else LineType.MOUNTAIN
        if listLines[i].targetTheta == -999: lineType = LineType.BORDER
        listLines_.append(Line(p1_index,p2_index,lineType,targetTheta))
    listPoints = [Point(x.position[0],x.position[2],x.position[1]) for x in listPoints]
    
    listFaces = get_faces_from_points_lines(listPoints, listLines_)
    o = OrigamiObject(listPoints, listLines_, listFaces)
    show_origami_object_2d_new(o,True,True)
    triangulate_all(listPoints,listLines_)
    
    listFaces = get_faces_from_points_lines(listPoints, listLines_)
    o = OrigamiObject(listPoints, listLines_, listFaces)
    show_origami_object_2d_new(o,True,True)
    show_origami_object_open3d(o,solverStep,30,True,True,True,True,2)

def show_full():
    IMAGE_PATH = "assets/M.svg"
    # listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listPoints, listLines = gen_ptu(np.pi/2,4,5)
    listLines_ = []
    for i in range(len(listLines)):
        p1_index = listPoints.index(listLines[i].p1)
        p1_index = listPoints.index(listLines[i].p2)
        targetTheta = torch.tensor(listLines[i].targetTheta)
        listLines_.append(Line(p1_index,p1_index,LineType.MOUNTAIN,targetTheta))
    listPoints = [Point(x.position[0],x.position[1],x.position[2]) for x in listPoints]
    listFaces = get_faces_from_points_lines(listPoints, listLines_)
    o = OrigamiObject(listPoints, listLines_, listFaces)
    setDeltaTime(o)
    # o.listPoints[0].is_fixed = True
    show_origami_object_open3d(o,solverStep,30,True,True,True,True,2)


# def show_full2():
#     IMAGE_PATH = "assets/M.svg"
#     listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)


#     listFaces = get_faces_from_points_lines(listPoints, listLines)
#     inputdict = convert_to_matrix(listPoints, listLines, listFaces)
#     ori = OrigamiObjectMatrix(inputdict["points"]/100.0,
#                           inputdict["lines"],
#                           inputdict["faces"],
#                           inputdict["target_thetas"],
#                           )
#     # o.listPoints[0].is_fixed = True
#     show_origami_object_open3d_ori_matrix(ori,30,True,True,True,True,2)

def main4():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object_2d_new(o,True,True)

def show_pointcloud():
    IMAGE_PATH = "assets/flappingBird.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    setDeltaTime(o)
    # o.listPoints[0].is_fixed = True
    plotter = Plotter(o, solverStep)
    plotter.show()

def benchmark():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    start_time = time.time()
    for i in range(1000):
        solverStep(o)
    print("time for 1000 steps of solverStep:", time.time() - start_time)

if __name__ == "__main__":
    import argparse
    import sys

    # 1. Create the parser
    parser = argparse.ArgumentParser(description="A simple file processor.")

    # 2. Add arguments
    parser.add_argument("function", help="function to call")

    # 3. Parse the arguments
    args = parser.parse_args()

    mapping = {
        'main': main,
        'main2': main2,
        'main_ptu': main_ptu,
        'show_full': show_full,
        # 'show_full2': show_full2,
        'main4': main4,
        'show_pointcloud': show_pointcloud,
        'benchmark': benchmark
    }

    # 4. Use the arguments in your script
    try:
        mapping[args.function]()

    except:
        print(f"Error: The function '{args.function}' was not found.")
        sys.exit(1)
