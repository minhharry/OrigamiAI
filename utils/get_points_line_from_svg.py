import lxml
from object.origami_object import Point, Line, LineType
from utils.snap_to_grid_svg import find_max_min
import matplotlib.pyplot as plt
from collections import defaultdict
from math import atan2


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

    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4):
        return Point(px, 0.0, py)
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

def create_points_lines(root) -> list[Point]:
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


 # Hàm để tính diện tích tam giác
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


def segments_intersect(p1, p2, q1, q2):
    """Kiểm tra 2 đoạn có cắt nhau (loại trừ trường hợp chung đỉnh)"""
    def orient(a, b, c):
        return (b.position[0] - a.position[0]) * (c.position[2] - a.position[2]) - \
               (b.position[2] - a.position[2]) * (c.position[0] - a.position[0])
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def find_polygons(listPoints, listLines):
    """Duyệt half-edge để tìm tất cả polygon"""
    adj = build_adjacency(listPoints, listLines)
    visited_half_edges = set()
    polygons = []

    for p in adj:
        for q in adj[p]:
            if (p, q) in visited_half_edges:
                continue

            polygon = [p]
            cur, prev = q, p
            while True:
                polygon.append(cur)
                visited_half_edges.add((prev, cur))
                neighbors = adj[cur]
                idx = neighbors.index(prev)
                nxt = neighbors[(idx - 1) % len(neighbors)]  # quay CCW
                prev, cur = cur, nxt
                if cur == polygon[0]:
                    break

            if len(polygon) > 2:
                m = min(polygon)
                mi = polygon.index(m)
                norm_poly = polygon[mi:] + polygon[:mi]
                if norm_poly not in polygons:
                    polygons.append(norm_poly)
    return polygons


def triangulate_polygon(polygon, listPoints, listLines, edges_set):
    """Dùng ear clipping để chia polygon thành tam giác"""
    def ccw(a, b, c):
        return (b.position[0] - a.position[0]) * (c.position[2] - a.position[2]) - \
               (b.position[2] - a.position[2]) * (c.position[0] - a.position[0]) > 0

    def is_valid_diagonal(i, j):
        if (min(i, j), max(i, j)) in edges_set:
            return False
        pi, pj = listPoints[i], listPoints[j]
        for k in range(len(polygon)):
            a, b = polygon[k], polygon[(k + 1) % len(polygon)]
            if len({i, j, a, b}) < 4:
                continue
            if segments_intersect(pi, pj, listPoints[a], listPoints[b]):
                return False
        return True

    poly = polygon[:]
    while len(poly) > 3:
        ear_found = False
        for i in range(len(poly)):
            prev_i = poly[(i - 1) % len(poly)]
            curr_i = poly[i]
            next_i = poly[(i + 1) % len(poly)]
            a, b, c = listPoints[prev_i], listPoints[curr_i], listPoints[next_i]

            if not ccw(a, b, c):
                continue
            if not is_valid_diagonal(prev_i, next_i):
                continue

            listLines.append(Line(prev_i, next_i, LineType.FACET))
            edges_set.add((min(prev_i, next_i), max(prev_i, next_i)))
            poly.pop(i)
            ear_found = True
            break

        if not ear_found:
            break
    return listLines


def triangulate_all(listPoints: list[Point], listLines: list[Line]) -> list[Line]:
    """Triangulate toàn bộ polygons phát hiện được"""
    edges_set = set((min(l.p1Index, l.p2Index), max(l.p1Index, l.p2Index)) for l in listLines)
    polygons = find_polygons(listPoints, listLines)
    for poly in polygons:
        listLines = triangulate_polygon(poly, listPoints, listLines, edges_set)
    return listLines   



if __name__ == "__main__":
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    print("Points: ",len(listPoints))
    print("Lines: ",len(listLines))
