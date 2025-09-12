from visualization.show_origami_object import show_origami_object, show_origami_object_2d
from object.origami_object import OrigamiObject, Point, Line, LineType

def main():
    listPoints = [Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9), Point(-10, 11, 12)]
    listLines = [
        Line(listPoints[0], listPoints[1], LineType.MOUNTAIN),
        Line(listPoints[1], listPoints[2], LineType.VALLEY),
        Line(listPoints[2], listPoints[3], LineType.BORDER),
        Line(listPoints[3], listPoints[0], LineType.FACET),
    ]
    o = OrigamiObject(listPoints, listLines)
    show_origami_object(o)
    show_origami_object_2d(o)


if __name__ == "__main__":
    main()
