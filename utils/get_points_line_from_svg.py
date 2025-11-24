import lxml.etree
from object.origami_object import Point, Line, LineType
from collections import defaultdict
from math import atan2
import torch
import numpy as np
from scipy.spatial import Delaunay
import math

IMAGE_PATH = "assets/flappingBird.svg"

NAMESPACE = '{http://www.w3.org/2000/svg}'
LINE_TAG = NAMESPACE + 'line'
RECT_TAG = NAMESPACE + 'rect'



import torch
import math

import math
from collections import defaultdict

def find_polygons(points, lines):
    import math
    from collections import defaultdict

    # Convert to 2D positions
    pos = [(float(p.position[0]), float(p.position[2])) for p in points]

    # Build adjacency
    adj = defaultdict(list)
    for L in lines:
        u, v = L.p1Index, L.p2Index
        adj[u].append(v)
        adj[v].append(u)

    # Sort neighbors of each vertex CCW
    def angle(u, v):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        return math.atan2(y2 - y1, x2 - x1)

    nbrs = {}
    for u in adj:
        nbrs[u] = sorted(adj[u], key=lambda v: angle(u, v))

    # Mark directed edges as unvisited
    visited = set()

    faces = []

    for u in adj:
        for v in adj[u]:
            if (u, v) in visited:
                continue

            face = []
            start = (u, v)
            cur_u, cur_v = u, v

            while True:
                visited.add((cur_u, cur_v))
                face.append(cur_u)

                nb = nbrs[cur_v]
                i = nb.index(cur_u)

                # FIX 1: turn-left = (i + 1)
                nxt = nb[(i + 1) % len(nb)]

                cur_u, cur_v = cur_v, nxt
                if (cur_u, cur_v) == start:
                    break

            if len(face) >= 3:
                # normalize
                m = min(range(len(face)), key=lambda i: face[i])
                f1 = face[m:] + face[:m]
                f2 = list(reversed(f1))
                faces.append(tuple(min(f1, f2)))

    faces = list(dict.fromkeys(faces))

    # compute area
    def poly_area(poly):
        a = 0
        for i in range(len(poly)):
            x1, y1 = pos[poly[i]]
            x2, y2 = pos[poly[(i + 1) % len(poly)]]
            a += x1*y2 - x2*y1
        return abs(a) * 0.5

    areas = [poly_area(f) for f in faces]
    if not areas:
        return []

    # remove outer face
    max_area = max(areas)
    faces = [faces[i] for i in range(len(faces)) if areas[i] != max_area]
    areas = [poly_area(f) for f in faces]

    # Helper: compute centroid
    def centroid(poly):
        xs = [pos[i][0] for i in poly]
        ys = [pos[i][1] for i in poly]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    # point-in-polygon
    def inside(pt, poly_xy):
        x, y = pt
        inside = False
        for i in range(len(poly_xy)):
            x1, y1 = poly_xy[i]
            x2, y2 = poly_xy[(i+1) % len(poly_xy)]
            if ((y1 > y) != (y2 > y)) and \
               (x < (x2-x1)*(y-y1)/(y2-y1) + x1):
                inside = not inside
        return inside

    faces_xy = [[pos[i] for i in f] for f in faces]

    valid = []
    for i, fi in enumerate(faces):
        Xi = faces_xy[i]
        Ai = areas[i]
        bad = False

        for j, fj in enumerate(faces):
            if i == j:
                continue

            Aj = areas[j]
            Xj_centroid = centroid(fj)

            if Aj < Ai and inside(Xj_centroid, Xi):
                bad = True
                break

        if not bad:
            valid.append(list(fi))

    return valid

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
    if line.attrib.get('stroke') == '#FF0000' or line.attrib.get('stroke') == '#ff0000':
        return LineType.MOUNTAIN
    elif line.attrib.get('stroke') == '#0000FF' or line.attrib.get('stroke') == '#0000ff':
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
            opacity = 1.0
            if child.attrib.get('opacity'):
                opacity = float(child.attrib.get('opacity')) 
            elif child.attrib.get('stroke-opacity'):
                opacity= float(child.attrib.get('stroke-opacity'))
            targetTheta = 0
            if (get_type_line_from_svg(child)==LineType.MOUNTAIN):
                targetTheta = -opacity*math.pi
            elif get_type_line_from_svg(child)==LineType.VALLEY:
                targetTheta = opacity*math.pi
            listLines.append(Line(index1, index2, get_type_line_from_svg(child),torch.tensor(targetTheta)))
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
                    newLine1 = Line(line.p1Index, i, line.lineType,line.targetTheta)
                    newLine2 = Line(i, line.p2Index, line.lineType,line.targetTheta)
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
        u1 = p1 - p2
        u2 = q1 - q2
        
        alpha = np.acos(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))
        
        if abs(alpha) < 0.01 or abs(alpha - np.pi) < 0.01:
            return True 
        else:
            # Nếu chúng chung đầu mút nhưng KHÔNG thẳng hàng (ví dụ: hình chữ L)
            return False

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
        new_lines.append(Line(i, j, LineType.FACET, torch.tensor(0.0)))

    return new_lines


from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

def calculate_polygon_area(listPoints: list[Point], listLines: list[Line], border_type_value=LineType.BORDER) -> float:
    """
    Tính diện tích đa giác dựa trên biên (boundary).
    
    Args:
        listPoints: Danh sách các object Point.
        listLines: Danh sách các object Line.
        border_type_value: Giá trị của lineType quy định là đường biên (màu đen).
                           Mặc định để là 1, bạn cần thay đổi tùy theo định nghĩa enum của bạn.
    Returns:
        Diện tích của đa giác.
    """
    
    # BƯỚC 1: Xây dựng đồ thị liên kết chỉ gồm các cạnh biên
    # Dictionary lưu các kết nối: {index_điểm: [các_index_hàng_xóm]}
    adj = defaultdict(list)
    
    boundary_lines_count = 0
    for line in listLines:
        # Cần check xem lineType có phải là enum hay int, ở đây so sánh giá trị
        if line.lineType == border_type_value:
            u, v = line.p1Index, line.p2Index
            adj[u].append(v)
            adj[v].append(u)
            boundary_lines_count += 1

    if boundary_lines_count < 3:
        print("Không đủ cạnh biên để tạo thành đa giác kín.")
        return 0.0

    # BƯỚC 2: Sắp xếp các đỉnh theo thứ tự vòng quanh (Graph Traversal)
    # Bắt đầu từ một điểm bất kỳ có trong danh sách biên
    start_node = list(adj.keys())[0]
    ordered_indices = [start_node]
    
    curr = start_node
    prev = -1 # Không có điểm trước đó cho điểm đầu tiên
    
    # Duyệt cho đến khi quay lại điểm đầu
    while True:
        neighbors = adj[curr]
        
        # Một đỉnh biên trong đa giác đơn phải có đúng 2 cạnh biên nối với nó
        if len(neighbors) != 2:
            print(f"Lỗi: Đỉnh {curr} không có đúng 2 cạnh biên (có {len(neighbors)}). Đồ thị không khép kín hoặc có lỗi.")
            return 0.0
        
        # Tìm điểm tiếp theo (không phải là điểm vừa đi qua)
        next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
        
        if next_node == start_node:
            break # Đã khép kín vòng
            
        ordered_indices.append(next_node)
        prev = curr
        curr = next_node

    # BƯỚC 3: Tính diện tích bằng công thức Shoelace (Shoelace Formula)
    # Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    
    area = 0.0
    n = len(ordered_indices)
    
    # Lấy tọa độ X, Y (bỏ qua Z vì tính diện tích mặt phẳng origami)
    # Chuyển sang list để truy xuất nhanh hoặc dùng tensor operation
    coords = []
    for idx in ordered_indices:
        pos = listPoints[idx].position
        coords.append((pos[0].item(), pos[1].item())) # Lấy x, y
    print("has ",len(coords))
    for i in range(n):
        x_i, y_i = coords[i]
        # Điểm tiếp theo (nếu là điểm cuối thì vòng về điểm đầu)
        x_next, y_next = coords[(i + 1) % n]
        
        area += (x_i * y_next) - (x_next * y_i)
        
    return 0.5 * abs(area)

def triangulate_polygons(listPoints: list, listLines: list, polygons: list[list[int]]) -> list[Line]:
    import numpy as np

    def angle_between(a, b, c):
        """Tính góc tại điểm b của tam giác a-b-c"""
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.arccos(cosang)

    all_added_edges = []

    for poly_indices in polygons:
        n = len(poly_indices)
        if n < 3:
            all_added_edges.append([])
            continue
        elif n == 3:
            all_added_edges.append([])
            continue

        # Lấy tọa độ
        pts = [listPoints[i].position.numpy() for i in poly_indices]

        # Tìm mặt phẳng và hệ trục 2D
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        u = v1 / np.linalg.norm(v1)
        v = np.cross(normal, u)

        # Chiếu xuống 2D
        pts2d = []
        for p in pts:
            vec = p - pts[0]
            x = np.dot(vec, u)
            y = np.dot(vec, v)
            pts2d.append(np.array([x, y]))

        indices2d = list(range(n))
        added_edges = []

        # helper: kiểm tra điểm trong tam giác
        def is_point_in_triangle(pt, a, b, c):
            dX = pt[0] - c[0]
            dY = pt[1] - c[1]
            dX21 = c[0] - b[0]
            dY12 = b[1] - c[1]
            D = dY12 * (a[0] - c[0]) + dX21 * (a[1] - c[1])
            s = dY12 * dX + dX21 * dY
            t = (c[1] - a[1]) * dX + (a[0] - c[0]) * dY
            if D < 0:
                return s <= 0 and t <= 0 and s + t >= D
            return s >= 0 and t >= 0 and s + t <= D

        while len(indices2d) > 3:
            best_ear = None
            best_angle = -1.0

            # duyệt mọi ear có thể
            for i in range(len(indices2d)):
                prev_idx = indices2d[i - 1]
                curr_idx = indices2d[i]
                next_idx = indices2d[(i + 1) % len(indices2d)]

                p_prev = pts2d[prev_idx]
                p_curr = pts2d[curr_idx]
                p_next = pts2d[next_idx]

                # kiểm tra convex
                cross = (p_curr[0]-p_prev[0])*(p_next[1]-p_curr[1]) - (p_curr[1]-p_prev[1])*(p_next[0]-p_curr[0])
                if cross <= 0:
                    continue

                # kiểm tra không có điểm nằm bên trong
                is_ear = True
                for j in indices2d:
                    if j in (prev_idx, curr_idx, next_idx):
                        continue
                    if is_point_in_triangle(pts2d[j], p_prev, p_curr, p_next):
                        is_ear = False
                        break
                if not is_ear:
                    continue

                # TÍNH GÓC LỚN NHẤT TẠI curr
                angle = angle_between(p_prev, p_curr, p_next)
                if angle > best_angle:
                    best_angle = angle
                    best_ear = (i, prev_idx, curr_idx, next_idx)

            if best_ear is None:
                print("Warning: cannot find ear – polygon may be non-simple")
                break

            # Lấy ear có góc lớn nhất
            i, prev_idx, curr_idx, next_idx = best_ear

            idx1 = poly_indices[prev_idx]
            idx2 = poly_indices[next_idx]

            # thêm cạnh mới
            added_edges.append(
                Line(idx1, idx2, LineType.FACET, torch.tensor(0.0))
            )

            # loại bỏ điểm ear
            indices2d.pop(i)

        all_added_edges.append(added_edges)

    # flatten
    return [edge for polygon_edges in all_added_edges for edge in polygon_edges]


def triangulate_all(listPoints: list[Point], listLines: list[Line]) -> list[Line]:
    init_listLines = listLines.copy()
    # listLines_ = triangluate_poly(listPoints, listLines)
    polygons = find_polygons(listPoints, init_listLines)
    listLines_ = triangulate_polygons(listPoints, listLines, polygons)
    
    print("polygons:",len(polygons))
    for poly in polygons:
        for i in range(len(poly)):
            print(poly[i])
        print("====")
    for line in listLines_:
        print(line)
        
    # for line in init_listLines:
    #     if line.lineType == LineType.BORDER:
    #         print(line)
    # print("Area before: ",calculate_polygon_area(listPoints, init_listLines))
    for line in listLines_:
        # if line.p1Index  in [0,1,2,3,4,5,6,7] and line.p2Index in [0,1,2,3,4,5,6,7]: continue
        listLines.append(line)
    print("OKKOKO")
    # print("Area after: ",calculate_polygon_area(listPoints, listLines))
    # listLines.append(Line(0,1,LineType.BORDER,torch.tensor(0.0)))
    # listLines.append(Line(1,2,LineType.BORDER,torch.tensor(0.0)))
    # listLines.append(Line(2,3,LineType.BORDER,torch.tensor(0.0)))
    # listLines.append(Line(3,0,LineType.BORDER,torch.tensor(0.0)))
    # listLines.append(Line(3,1,LineType.BORDER,torch.tensor(0.0)))

    return listLines 




if __name__ == "__main__":
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    print("Points: ",len(listPoints))
    print("Lines: ",len(listLines))
