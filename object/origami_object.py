import torch
import math
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
    
class LineType(Enum):
    MOUNTAIN = 1
    VALLEY = 2
    BORDER = 3
    FACET = 4


class Line:
    def __init__(self, p1Index: int, p2Index: int, lineType: LineType, targetTheta: float = 0) -> None:
        self.p1Index = p1Index
        self.p2Index = p2Index
        self.lineType = lineType
        self.targetTheta = targetTheta
        self.currentTheta = 0.0
        self.lastTheta = 0.0

    def __str__(self) -> str:
        return f"Line({self.p1Index}, {self.p2Index}, {self.lineType},{self.targetTheta},{self.currentTheta} )"

    def __repr__(self) -> str:
        return str(self)

    def get_line_original_length(self,listPoints: list[Point])->float:
        return torch.linalg.norm(listPoints[self.p1Index].originalPosition-listPoints[self.p2Index].originalPosition)
    def get_line_length(self,listPoints: list[Point])->float:
        return torch.linalg.norm(listPoints[self.p1Index].position-listPoints[self.p2Index].position)
    

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

    def calculate_and_update_normal(self, listPoints: list[Point]) -> torch.Tensor:
        p1 = listPoints[self.point1Index].position
        p2 = listPoints[self.point2Index].position
        p3 = listPoints[self.point3Index].position
        v1 = p2 - p1
        v2 = p3 - p1
     
        normal = torch.linalg.cross(v1, v2)
        normal = normal / torch.linalg.norm(normal)
        self.normal = normal
        return normal

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
    
class OrigamiObject:
    def __init__(
        self,
        listPoints: list[Point],
        listLines: list[Line],
        listFaces: list[Face] = [],
    ) -> None:
        self.listPoints: list[Point] = listPoints  # list of points, order is not change

        min_position_x = 9999999
        max_position_x = -9999999
        min_position_y = 9999999
        max_position_y = -9999999
        a = -1.0/math.sqrt(2)
        b = -a
        for point in listPoints:
            min_position_x = min(point.originalPosition.tolist()[0],min_position_x)
            max_position_x = max(point.originalPosition.tolist()[0],max_position_x)
            min_position_y = min(point.originalPosition.tolist()[2],min_position_y)
            max_position_y = max(point.originalPosition.tolist()[2],max_position_y)
            
        #Doan nay thenm vo khong thay cai hinh luon
        for point in listPoints:
            x_ = a + (point.originalPosition.tolist()[0]-min_position_x)*(b-a)/(max_position_x-min_position_x)
            z_ = a + (point.originalPosition.tolist()[2]-min_position_y)*(b-a)/(max_position_y-min_position_y)
            point.originalPosition = torch.tensor([x_,0.0,z_])
            point.position = point.originalPosition.clone()

        self.listLines: list[Line] = listLines 
        self.listFaces: list[Face] = listFaces 
        self.graph: list[list[tuple[int, int]]] = [
            [] for _ in range(len(listPoints))
        ]  # tuple of (point index, line index)
        for line_index in range(len(listLines)):
            p1Index, p2Index = listLines[line_index].p1Index, listLines[line_index].p2Index
            self.graph[p1Index].append((p2Index, line_index))
            self.graph[p2Index].append((p1Index, line_index))
        
        for face in listFaces:
            face.alpha1, face.alpha2, face.alpha3 = \
                OrigamiObject.calculate_face_angles(
                    listPoints[face.point1Index], listPoints[face.point2Index], listPoints[face.point3Index]
                    )
            
            face.calculate_and_update_line_index(listLines)

        self.mappingLineToFace: dict[int, list[int]] = {} #index cạnh
        for face_index in range(len(self.listFaces)):
            face = self.listFaces[face_index]
            if self.mappingLineToFace.get(face.line12Index) is None:
                self.mappingLineToFace[face.line12Index] = []
            if self.mappingLineToFace.get(face.line23Index) is None:
                self.mappingLineToFace[face.line23Index] = []
            if self.mappingLineToFace.get(face.line13Index) is None:
                self.mappingLineToFace[face.line13Index] = []
            self.mappingLineToFace[face.line12Index].append(face_index)
            self.mappingLineToFace[face.line23Index].append(face_index)
            self.mappingLineToFace[face.line13Index].append(face_index)
      
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
    def calculate_distance_point_to_line(cls, point: Point, line: Line, points: list[Point]) -> torch.Tensor:
        A = points[line.p1Index].position
        B = points[line.p2Index].position
        P = point.position
        numerator = torch.norm(torch.linalg.cross(P - A, P - B))
        denominator = torch.norm(B - A)
        return numerator / denominator

    @classmethod
    def calculate_distance_point_to_line_2(cls, point: Point, line_point_1: Point, line_point_2: Point) -> torch.Tensor:
        A = line_point_1.position
        B = line_point_2.position
        P = point.position
        numerator = torch.norm(torch.linalg.cross(P - A, P - B))
        denominator = torch.norm(B - A)
        return numerator / denominator

    def calculate_theta(self, lineIndex: int, p3: Point, p4: Point,face1: Face, face2: Face, fold_percent = 1.0) -> float:
        if len(self.mappingLineToFace[lineIndex]) != 2:    return 0.0
        
        n1 = face1.calculate_and_update_normal(self.listPoints)
        n2 = face2.calculate_and_update_normal(self.listPoints)
        
        node0 = p3.position
        node1 = p4.position
        creaseVector = torch.nn.functional.normalize(node1 - node0, dim=0)
        
        dotNormals = torch.dot(n1, n2).clamp(-1.0, 1.0)
        
        cross_n1_crease = torch.cross(n1, n2)
        y = torch.dot(cross_n1_crease, creaseVector)
        
        unsignedTheta = torch.acos(dotNormals)  
        
        signTheta = torch.sign(y)
        
        theta = unsignedTheta * signTheta
        
        lastTheta = self.listLines[lineIndex].lastTheta
        diff = theta - lastTheta
        TWO_PI = 2 * math.pi
        if diff < -5.0:
            diff += TWO_PI
        elif diff > 5.0:
            diff -= TWO_PI
        theta = lastTheta + diff
        
        self.listLines[lineIndex].lastTheta = theta.item()
        
        return theta.item()
    
    @classmethod
    def calculate_face_angles(cls, p1: Point, p2: Point, p3: Point) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
