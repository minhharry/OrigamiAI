from visualization.show_origami_object import show_origami_object, show_origami_object_2d, show_origami_object_2d_new, show_faces_2d
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from physic_engine.calculate_normals import calculate_and_update_normals
from utils.get_points_line_from_svg import get_points_line_from_svg
from utils.get_faces_from_points_lines import get_faces_from_points_lines

def main():
    IMAGE_PATH = "assets/flappingBird.svg"
    # listPoints = [Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9), Point(-10, 11, 12)]
    # listLines = [
    #     Line(0, 1, LineType.MOUNTAIN),
    #     Line(1, 2, LineType.VALLEY),
    #     Line(2, 3, LineType.BORDER),
    #     Line(3, 0, LineType.FACET),
    #     Line(1, 3, LineType.FACET),
    # ]
    # listFaces = [Face(1, 2, 3), Face(0, 1, 3)]
    # o = OrigamiObject(listPoints, listLines,listFaces)
    # calculate_and_update_normals(o)
    # show_origami_object(o)
    # show_origami_object_2d(o)

    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    calculate_and_update_normals(o)
    show_origami_object(o)
    show_origami_object_2d_new(o)
    show_faces_2d(o)


if __name__ == "__main__":
    main()
