import numpy as np
from enum import Enum


class Point:
    def __init__(self, x: int, y: int, z: int) -> None:
        self.position = np.array([x, y, z])
        self.originalPosition = np.array([x, y, z])

    def __str__(self):
        return f"Point({self.position[0]}, {self.position[1]}, {self.position[2]})"

    def __repr__(self) -> str:
        return str(self)


class LineType(Enum):
    MOUNTAIN = 1
    VALLEY = 2
    BORDER = 3
    FACET = 4


class Line:
    def __init__(self, p1: Point, p2: Point, lineType: LineType) -> None:
        self.p1 = p1
        self.p2 = p2
        self.lineType = lineType

    def __str__(self) -> str:
        return f"Line({self.p1}, {self.p2})"

    def __repr__(self) -> str:
        return str(self)


class OrigamiObject:
    def __init__(
        self,
        listPoints: list[Point],
        listLines: list[Line],
    ) -> None:
        self.listPoints: list[Point] = listPoints  # list of points, order is not change
        self.listLines: list[Line] = listLines  # list of lines
        self.graph: list[list[tuple[int, int]]] = [
            [] for _ in range(len(listPoints))
        ]  # tuple of (point index, line index)
        for line_index in range(len(listLines)):
            p1, p2 = (
                listLines[line_index].p1.originalPosition,
                listLines[line_index].p2.originalPosition,
            )
            p1Index, p2Index = -1, -1
            for point_index in range(len(listPoints)):
                if listPoints[point_index].originalPosition.all() == p1.all():
                    p1Index = point_index
                if listPoints[point_index].originalPosition.all() == p2.all():
                    p2Index = point_index
                if p1Index == -1 or p2Index == -1:
                    raise Exception("Point not found")
                self.graph[p1Index].append((p2Index, line_index))
                self.graph[p2Index].append((p1Index, line_index))

    def __str__(self) -> str:
        return f"OrigamiObject({self.listPoints}, {self.listLines})"

    def __repr__(self) -> str:
        return str(self)


if __name__ == "__main__":
    listPoints = [Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9), Point(10, 11, 12)]
    listLines = [
        Line(listPoints[0], listPoints[1], LineType.MOUNTAIN),
        Line(listPoints[1], listPoints[2], LineType.VALLEY),
        Line(listPoints[2], listPoints[3], LineType.BORDER),
        Line(listPoints[3], listPoints[0], LineType.FACET),
    ]
    o = OrigamiObject(listPoints, listLines)
    print(o)
    print(o.graph)
    print(LineType.MOUNTAIN)