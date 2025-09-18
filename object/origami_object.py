import torch
from enum import Enum

class Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.position = torch.tensor([x, y, z], dtype=torch.float32)
        self.originalPosition = torch.tensor([x, y, z], dtype=torch.float32)
        self.force = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.verlocity = torch.tensor([0, 0, 0], dtype=torch.float32)

    def __str__(self):
        return f"Point({self.position[0]}, {self.position[1]}, {self.position[2]})"

    def __repr__(self) -> str:
        return str(self)
    
    def clear(self) -> None:
        self.force = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.verlocity = torch.tensor([0, 0, 0], dtype=torch.float32)
    
class LineType(Enum):
    MOUNTAIN = 1
    VALLEY = 2
    BORDER = 3
    FACET = 4


class Line:
    def __init__(self, p1Index: int, p2Index: int, lineType: LineType) -> None:
        self.p1Index = p1Index
        self.p2Index = p2Index
        self.lineType = lineType

    def __str__(self) -> str:
        return f"Line({self.p1Index}, {self.p2Index}, {self.lineType})"

    def __repr__(self) -> str:
        return str(self)

class Face:
    def __init__(self, point1Index: int, point2Index: int, point3Index: int) -> None:
        self.point1Index = point1Index
        self.point2Index = point2Index
        self.point3Index = point3Index
        
        self.line12Index = -1
        self.line23Index = -1
        self.line13Index = -1

        self.normal = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.alpha1 = torch.tensor(0.0) # Angle 213
        self.alpha2 = torch.tensor(0.0) # Angle 123
        self.alpha3 = torch.tensor(0.0) # Angle 132

    def __str__(self) -> str:
        return f"Face({self.point1Index}, {self.point2Index}, {self.point3Index})"

    def __repr__(self) -> str:
        return str(self)

    def calculate_and_update_normal(self, listPoints: list[Point]) -> None:
        p1 = listPoints[self.point1Index].position
        p2 = listPoints[self.point2Index].position
        p3 = listPoints[self.point3Index].position
        v1 = p2 - p1
        v2 = p3 - p1
        normal = torch.linalg.cross(v1, v2)
        normal = normal / torch.linalg.norm(normal)
        self.normal = normal

    def calculate_and_update_line_index(self, listLines: list[Line]) -> None:
        def find_line_index(p1Index: int, p2Index: int) -> int:
            for i in range(len(listLines)):
                if listLines[i].p1Index == p1Index and listLines[i].p2Index == p2Index:
                    return i
                if listLines[i].p1Index == p2Index and listLines[i].p2Index == p1Index:
                    return i
            return -1
        self.line12Index = find_line_index(self.point1Index, self.point2Index)
        self.line23Index = find_line_index(self.point2Index, self.point3Index)
        self.line13Index = find_line_index(self.point1Index, self.point3Index)

    def calculate_face_angles(self, p1: Point, p2: Point, p3: Point) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = p1.position
        b = p2.position
        c = p3.position
        def angle(u, v):
            cos_theta = torch.dot(u, v) / (torch.norm(u) * torch.norm(v))
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            return torch.acos(cos_theta)
        
        ab, ac = b - a, c - a
        ba, bc = a - b, c - b
        ca, cb = a - c, b - c
        
        angle_A = angle(ab, ac)
        angle_B = angle(ba, bc)
        angle_C = angle(ca, cb)
        return angle_A, angle_B, angle_C

class OrigamiObject:
    def __init__(
        self,
        listPoints: list[Point],
        listLines: list[Line],
        listFaces: list[Face] = [],
    ) -> None:
        self.listPoints: list[Point] = listPoints  # list of points, order is not change
        self.listLines: list[Line] = listLines  # list of lines
        self.listFaces: list[Face] = listFaces  # list of faces
        self.graph: list[list[tuple[int, int]]] = [
            [] for _ in range(len(listPoints))
        ]  # tuple of (point index, line index)
        for line_index in range(len(listLines)):
            p1Index, p2Index = listLines[line_index].p1Index, listLines[line_index].p2Index
            self.graph[p1Index].append((p2Index, line_index))
            self.graph[p2Index].append((p1Index, line_index))
        
        # Calculate face original angles
        for face in listFaces:
            face.alpha1, face.alpha2, face.alpha3 = \
                face.calculate_face_angles(listPoints[face.point1Index], listPoints[face.point2Index], listPoints[face.point3Index])
            
            face.calculate_and_update_line_index(listLines)

    def __str__(self) -> str:
        return f"OrigamiObject({self.listPoints}, {self.listLines})"

    def __repr__(self) -> str:
        return str(self)
    
    @classmethod
    def getDistance(cls, point: Point, otherPoint: Point) -> torch.Tensor:
        return torch.linalg.norm(point.position - otherPoint.position)
    
    @classmethod
    def getOriginalDistance(cls, point: Point, otherPoint: Point) -> torch.Tensor:
        return torch.linalg.norm(point.originalPosition - otherPoint.originalPosition)
    
    @classmethod
    def getUnitVector(cls, point1: Point, point2: Point) -> torch.Tensor:
        return (point2.position - point1.position) / torch.linalg.norm(point2.position - point1.position)
    
    @classmethod
    def distance_point_to_line(cls, point: Point, line: Line, points: list[Point]) -> torch.Tensor:
        A = points[line.p1Index].position
        B = points[line.p2Index].position
        P = point.position
        numerator = torch.norm(torch.linalg.cross(P - A, P - B))
        denominator = torch.norm(B - A)
        return numerator / denominator
    

if __name__ == "__main__":
    listPoints = [Point(0,1,2), Point(4, 5, 6), Point(7, 8, 9), Point(10, 11, 12)]
    listLines = [
        Line(0, 1, LineType.MOUNTAIN),
        Line(1, 1, LineType.VALLEY),
        Line(2, 3, LineType.BORDER),
        Line(3, 0, LineType.FACET),
    ]
    o = OrigamiObject(listPoints, listLines)
    print(o)
    print(o.graph)
    print(LineType.MOUNTAIN)

    listPoints2 = [Point(0,3,0), Point(4,0,0), Point(0,0,0)]
    face = Face(0, 1, 2)
    print(face.calculate_face_angles(listPoints2[0], listPoints2[1], listPoints2[2]))