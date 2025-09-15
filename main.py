from visualization.show_origami_object import show_origami_object, show_origami_object_2d, show_origami_object_2d_new, show_faces_2d
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from utils.get_points_line_from_svg import get_points_line_from_svg
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from physic_engine.solver import solver

def main():
    IMAGE_PATH = "assets/flappingBird.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object(o)
    show_origami_object_2d(o)
    show_faces_2d(o)

def main2():
    IMAGE_PATH = "assets/flappingBird.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    print("points:", len(o.listPoints))
    print("lines:", len(o.listLines))
    print("faces:", len(o.listFaces))
    solver(o)


if __name__ == "__main__":
    main2()
