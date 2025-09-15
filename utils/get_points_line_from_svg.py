import lxml.etree
from object.origami_object import Point, Line, LineType
from collections import defaultdict
from math import atan2
import torch
import numpy as np
from scipy.spatial import Delaunay


IMAGE_PATH = "assets/flappingBird.svg"

NAMESPACE = '{http://www.w3.org/2000/svg}'
LINE_TAG = NAMESPACE + 'line'
RECT_TAG = NAMESPACE + 'rect'

def get_intersection_point(listPoints: list[Point],line1: Line, line2: Line) -> Point | None:
    x1, y1 = listPoints[line1.p1Index].position[0], listPoints[line1.p1Index].position[2]
    x2, y2 = listPoints[line1.p2Index].position[0], listPoints[line1.p2Index].position[2]
    x3, y3 = listPoints[line2.p1Index].position[0], listPoints[line2.p1Index].position[2]
    x4, y4 = listPoints[line2.p2Index].position[0], listPoints[line2.p2Index].position[2]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None 

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    if torch.min(x1, x2) <= px <= torch.max(x1, x2) and torch.min(y1, y2) <= py <= torch.max(y1, y2) and torch.min(x3, x4) <= px <= torch.max(x3, x4) and torch.min(y3, y4) <= py <= torch.max(y3, y4):
        return Point(float(px), 0.0, float(py))
    return None

def is_point_exist(point: Point, listPoints: list[Point], pointMergeTolerance: float = 3.0) -> bool:
    for p in listPoints:
        if abs(p.position[0] - point.position[0]) < pointMergeTolerance and abs(p.position[1] - point.position[1]) < pointMergeTolerance and abs(p.position[2] - point.position[2]) < pointMergeTolerance:
            return True
    return False

def is_line_exist(line: Line, listLines: list[Line]) -> bool:
    for l in listLines:
        if (l.p1Index == line.p1Index and l.p2Index == line.p2Index) or (l.p1Index == line.p2Index and l.p2Index == line.p1Index):
            return True
    return False

def find_point_index(point: Point, listPoints: list[Point], pointMergeTolerance: float = 3.0) -> int:
    for i, p in enumerate(listPoints):
        if abs(p.position[0] - point.position[0]) < pointMergeTolerance and abs(p.position[1] - point.position[1]) < pointMergeTolerance and abs(p.position[2] - point.position[2]) < pointMergeTolerance:
            return i
    return -1

def is_on_line(point: Point, line: Line, listPoints: list[Point], pointMergeTolerance: float = 3.0) -> bool:
    p1 = listPoints[line.p1Index]
    p2 = listPoints[line.p2Index]
    cross_product = (point.position[2] - p1.position[2]) * (p2.position[0] - p1.position[0]) - (point.position[0] - p1.position[0]) * (p2.position[2] - p1.position[2])
    if abs(cross_product) > pointMergeTolerance:
        return False

    dot_product = (point.position[0] - p1.position[0]) * (p2.position[0] - p1.position[0]) + (point.position[2] - p1.position[2]) * (p2.position[2] - p1.position[2])
    if dot_product < 0:
        return False

    squared_length_p1p2 = (p2.position[0] - p1.position[0]) ** 2 + (p2.position[2] - p1.position[2]) ** 2
    if dot_product > squared_length_p1p2:
        return False

    return True

def get_type_line_from_svg(line) -> LineType:
    if line.attrib.get('stroke') == '#FF0000':
        return LineType.MOUNTAIN
    elif line.attrib.get('stroke') == '#0000FF':
        return LineType.VALLEY
    elif line.attrib.get('stroke') == '#000000':
        return LineType.BORDER
    else:
        return LineType.FACET

def create_points_lines(root) -> tuple[list[Point], list[Line]]:
    listPoints: list[Point] = []
    listLines: list[Line] = []
    for child in root:
        if child.tag == RECT_TAG:
            listPoints.append(Point(float(child.attrib['x']),0.0,float(child.attrib['y'])))
            listPoints.append(Point(float(child.attrib['x'])+float(child.attrib['width']),0.0,float(child.attrib['y'])))
            listPoints.append(Point(float(child.attrib['x'])+float(child.attrib['width']),0.0,float(child.attrib['y'])+float(child.attrib['height'])))
            listPoints.append(Point(float(child.attrib['x']),0.0,float(child.attrib['y'])+float(child.attrib['height'])))
            listLines.append(Line(0, 1, LineType.BORDER))
            listLines.append(Line(1, 2, LineType.BORDER))
            listLines.append(Line(2, 3, LineType.BORDER))
            listLines.append(Line(3, 0, LineType.BORDER))
        if child.tag == LINE_TAG:
            tempPoint1 = Point(float(child.attrib['x1']),0.0,float(child.attrib['y1']))
            tempPoint2 = Point(float(child.attrib['x2']),0.0,float(child.attrib['y2']))
            if not is_point_exist(tempPoint1, listPoints):
                listPoints.append(tempPoint1)
            if not is_point_exist(tempPoint2, listPoints):
                listPoints.append(tempPoint2)
            index1 = find_point_index(tempPoint1, listPoints)
            index2 = find_point_index(tempPoint2, listPoints)
            listLines.append(Line(index1, index2, get_type_line_from_svg(child)))
        for i in range(len(listLines)):
            for j in range(i+1, len(listLines)):
                intersectionPoint = get_intersection_point(listPoints,listLines[i], listLines[j])
                if intersectionPoint is not None and not is_point_exist(intersectionPoint, listPoints):
                    listPoints.append(intersectionPoint)
    return listPoints, listLines

def break_lines(listPoints: list[Point], listLines: list[Line]) -> list[Line]:

    #break line if have point on line
    for line in listLines:
        for i, point in enumerate(listPoints):
            if is_on_line(point, line, listPoints):
                if i != line.p1Index and i != line.p2Index:
                    newLine1 = Line(line.p1Index, i, line.lineType)
                    newLine2 = Line(i, line.p2Index, line.lineType)
                    if not is_line_exist(newLine1, listLines):
                        listLines.append(newLine1)
                    if not is_line_exist(newLine2, listLines):
                        listLines.append(newLine2)
                    line.p1Index=-1
                    line.p2Index=-1
                    break
    listLines = [line for line in listLines if line.p1Index != -1 and line.p2Index != -1]
    return listLines

def get_points_line_from_svg(svg_file_path: str) -> tuple[list[Point], list[Line]]:
    root = lxml.etree.parse(svg_file_path).getroot()
    listPoints, listLines = create_points_lines(root) #get all points , include intersection points
    listLines = break_lines(listPoints, listLines) # break lines if have point on line
    triangulate_all(listPoints, listLines) # triangulate all polygon to facet lines
    return listPoints, listLines

 

def build_adjacency(listPoints, listLines):
    """Tạo adjacency list, các neighbor được sắp CCW quanh mỗi điểm"""
    adj = defaultdict(list)
    for line in listLines:
        p1, p2 = line.p1Index, line.p2Index
        adj[p1].append(p2)
        adj[p2].append(p1)

    for p, neighs in adj.items():
        x0, y0 = listPoints[p].position[0], listPoints[p].position[2]
        adj[p] = sorted(neighs, key=lambda q: atan2(
            listPoints[q].position[2] - y0,
            listPoints[q].position[0] - x0
        ))
    return adj

def do_segments_intersect(p1, p2, q1, q2) -> bool:
    """Kiểm tra 2 đoạn thẳng (p1,p2) và (q1,q2) có cắt nhau không (trừ khi trùng endpoint)."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

    if (p1 == q1).all() or (p1 == q2).all() or (p2 == q1).all() or (p2 == q2).all():
        return False  # chia sẻ endpoint thì không tính là giao cắt

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def triangluate_poly(listPoints: list[Point], listLines: list[Line]) -> list[Line]:
    """
    Từ listPoints sinh ra thêm các cạnh FACET bằng Delaunay.
    Không đè lên cạnh có sẵn và không cắt biên polygon.
    """

    # chọn 2 chiều có biến thiên lớn nhất
    arr = np.array([p.position.numpy() for p in listPoints])
    ranges = arr.max(axis=0) - arr.min(axis=0)
    idx = ranges.argsort()[-2:]
    points_2d = arr[:, idx]

    tri = Delaunay(points_2d)

    # tập hợp cạnh từ simplex
    edges = set()
    for simplex in tri.simplices:
        i, j, k = simplex
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    # các cạnh có sẵn
    existing_edges = {(min(l.p1Index, l.p2Index), max(l.p1Index, l.p2Index)) for l in listLines}
    existing_segments = [(points_2d[i], points_2d[j]) for (i, j) in existing_edges]

    new_lines = []
    for i, j in edges:
        if (i, j) in existing_edges:
            continue  # bỏ qua cạnh đã có

        pi, pj = points_2d[i], points_2d[j]

        # kiểm tra cắt cạnh biên
        intersects = False
        for q1, q2 in existing_segments:
            if do_segments_intersect(pi, pj, q1, q2):
                intersects = True
                break
        if intersects:
            continue
        new_lines.append(Line(i, j, LineType.FACET))

    return new_lines


def triangulate_all(listPoints: list[Point], listLines: list[Line]) -> list[Line]:
    listLines_ = triangluate_poly(listPoints, listLines)
    for line in listLines_:
        listLines.append(line)
    return listLines 



if __name__ == "__main__":
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    print("Points: ",len(listPoints))
    print("Lines: ",len(listLines))
