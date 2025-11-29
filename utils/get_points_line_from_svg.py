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

    # ---- convert to 2D ----
    pos = [(float(p.position[0]), float(p.position[2])) for p in points]

    # ---- build raw edges ----
    edges = []
    for L in lines:
        u = L.p1Index
        v = L.p2Index
        edges.append((u, v))
        edges.append((v, u))  # half-edge

    # ---- group half-edges by origin ----
    outgoing = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        outgoing[u].append((v, i))

    # ---- sort outgoing edges CCW for each vertex ----
    def angle(u, v):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        return math.atan2(y2 - y1, x2 - x1)

    for u in outgoing:
        outgoing[u].sort(key=lambda x: angle(u, x[0]))

    # ---- build "next" pointer for each half-edge ----
    next_edge = [None] * len(edges)

    for u in outgoing:
        nbrs = outgoing[u]  # list of (v, eid)
        deg = len(nbrs)

        for i in range(deg):
            v, eid_uv = nbrs[i]

            # twin is (v -> u)
            # find in outgoing[v] the entry whose destination == u
            lst_v = outgoing[v]
            j = next(k for k in range(len(lst_v)) if lst_v[k][0] == u)

            # next half-edge is the CCW next around vertex v
            v_next, eid_vp = lst_v[(j - 1) % len(lst_v)]
            next_edge[eid_uv] = eid_vp

    # ---- walk faces ----
    visited = [False] * len(edges)
    faces = []

    for eid in range(len(edges)):
        if visited[eid]:
            continue

        face = []
        cur = eid

        while not visited[cur]:
            visited[cur] = True
            u, v = edges[cur]
            face.append(u)
            cur = next_edge[cur]

        if len(face) >= 3:
            # canonical form
            m = min(range(len(face)), key=lambda i: face[i])
            f1 = tuple(face[m:] + face[:m])
            f2 = tuple(reversed(f1))
            faces.append(min(f1, f2))

    # ---- remove duplicates ----
    faces = list(dict.fromkeys(faces))

    # ---- compute area ----
    def area(f):
        a = 0
        for i in range(len(f)):
            x1,y1 = pos[f[i]]
            x2,y2 = pos[f[(i+1)%len(f)]]
            a += x1*y2 - x2*y1
        return abs(a)/2

    # ---- remove outer face ----
    A = [area(f) for f in faces]
    maxA = max(A)
    faces = [faces[i] for i in range(len(faces)) if A[i] != maxA]

    return [list(f) for f in faces]

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
def triangulate_polygons(listPoints: list, listLines: list, polygons: list[list[int]]) -> list:
    import numpy as np
    import torch
    EPS = 1e-9

    def to2d(pt):
        # dùng (x, z) như bạn đã dùng trước
        return np.array([float(pt.position[0]), float(pt.position[2])], dtype=float)

    # existing edges set (unordered pairs)
    existing_edges = set()
    for ln in listLines:
        a, b = int(ln.p1Index), int(ln.p2Index)
        existing_edges.add((min(a, b), max(a, b)))

    added_edges = []

    # ---- helper geometry ----
    def orient(a, b, c):
        # orientation (signed area *2) of triangle a-b-c
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, p):
        # check p on segment a-b (collinear assumed)
        return min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS and \
               min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS

    def segments_intersect_strict(a,b,c,d):
        # return True if segments ab and cd intersect (including collinear overlap)
        o1 = orient(a,b,c)
        o2 = orient(a,b,d)
        o3 = orient(c,d,a)
        o4 = orient(c,d,b)

        if abs(o1) < EPS and on_segment(a,b,c): return True
        if abs(o2) < EPS and on_segment(a,b,d): return True
        if abs(o3) < EPS and on_segment(c,d,a): return True
        if abs(o4) < EPS and on_segment(c,d,b): return True

        return (o1>0 and o2<0 or o1<0 and o2>0) and (o3>0 and o4<0 or o3<0 and o4>0)

    def point_in_poly(pt, poly):
        # ray casting robust
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            # check edge on point
            if abs(orient((x1,y1),(x2,y2),(x,y))) < EPS and on_segment((x1,y1),(x2,y2),(x,y)):
                return True
            intersect = ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1 + 1e-20) + x1)
            if intersect:
                inside = not inside
        return inside

    def point_in_triangle(p, a, b, c):
        # barycentric / signs
        o1 = orient(a,b,p)
        o2 = orient(b,c,p)
        o3 = orient(c,a,p)
        # allow on-edge
        return (o1 >= -EPS and o2 >= -EPS and o3 >= -EPS) or (o1 <= EPS and o2 <= EPS and o3 <= EPS)

    # ---- main loop per polygon ----
    for poly_indices in polygons:
        m = len(poly_indices)
        if m < 3:
            continue
        if m == 3:
            continue  # nothing to add

        pts2d = [to2d(listPoints[i]) for i in poly_indices]
        # polygon orientation (signed area)
        signed_area = 0.0
        for i in range(m):
            x1,y1 = pts2d[i]
            x2,y2 = pts2d[(i+1)%m]
            signed_area += (x1*y2 - x2*y1)
        ccw = signed_area > 0  # True if CCW orientation

        # working index list (local indices into pts2d / poly_indices)
        idxs = list(range(m))
        local_added = []

        # quick access edges of polygon in local-index terms
        def poly_edge_local(i):
            return (idxs[i], idxs[(i+1)%len(idxs)])

        # try ear clipping
        safe_guard = 0
        while len(idxs) > 3 and safe_guard < 5*m:
            safe_guard += 1
            ear_found = False
            L = len(idxs)
            for k in range(L):
                i_prev = idxs[(k-1)%L]
                i_curr = idxs[k]
                i_next = idxs[(k+1)%L]

                A = pts2d[i_prev]; B = pts2d[i_curr]; C = pts2d[i_next]

                cross = orient(A,B,C)
                # convex test depends on orientation
                if ccw:
                    if cross <= EPS:
                        continue
                else:
                    if cross >= -EPS:
                        continue

                # no other vertex inside triangle
                any_inside = False
                for j in idxs:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if point_in_triangle(pts2d[j], A, B, C):
                        any_inside = True
                        break
                if any_inside:
                    continue

                # prospective diagonal is (i_prev, i_next) in local indices
                g1 = poly_indices[i_prev]
                g2 = poly_indices[i_next]
                # skip if diagonal equals existing polygon edge or global existing edge
                if tuple(sorted((g1, g2))) in existing_edges:
                    # if it's polygon boundary (adjacent in original polygon) it's allowed only if it's the boundary edge (shouldn't be)
                    pass

                # check diagonal doesn't intersect polygon edges (except at endpoints)
                p1 = pts2d[i_prev]; p2 = pts2d[i_next]
                intersects = False
                for t in range(len(idxs)):
                    j1 = idxs[t]; j2 = idxs[(t+1)%len(idxs)]
                    # skip edges touching at endpoints
                    if j1 in (i_prev, i_next) or j2 in (i_prev, i_next):
                        continue
                    q1 = pts2d[j1]; q2 = pts2d[j2]
                    if segments_intersect_strict(p1,p2,q1,q2):
                        intersects = True
                        break
                if intersects:
                    continue

                # diagonal midpoint must be inside polygon (not outside)
                mid = (p1 + p2) * 0.5
                full_poly_xy = [pts2d[i] for i in range(len(pts2d))]
                if not point_in_poly(mid, full_poly_xy):
                    continue

                # good ear -> add diagonal as triangle edge between global indices
                new_line = Line(int(g1), int(g2), LineType.FACET, torch.tensor(0.0))
                local_added.append(new_line)
                existing_edges.add(tuple(sorted((int(g1), int(g2)))))

                # remove ear vertex (i_curr) from polygon
                idxs.pop(k)
                ear_found = True
                break

            if not ear_found:
                # no ear found -> possibly non-simple or numerical problem
                # try fallback: pick any diagonal that does not intersect and whose midpoint inside
                fallback_done = False
                for a_local in range(len(idxs)):
                    for b_local in range(a_local+2, len(idxs)):
                        # avoid adjacent and wrap-around adjacency
                        if (b_local == a_local+1) or (a_local==0 and b_local==len(idxs)-1):
                            continue
                        i1 = idxs[a_local]; i2 = idxs[b_local]
                        g1 = poly_indices[i1]; g2 = poly_indices[i2]
                        if tuple(sorted((g1,g2))) in existing_edges:
                            continue
                        p1 = pts2d[i1]; p2 = pts2d[i2]
                        intersects = False
                        for t in range(len(idxs)):
                            j1 = idxs[t]; j2 = idxs[(t+1)%len(idxs)]
                            if j1 in (i1,i2) or j2 in (i1,i2):
                                continue
                            if segments_intersect_strict(p1,p2, pts2d[j1], pts2d[j2]):
                                intersects = True
                                break
                        if intersects:
                            continue
                        mid = (p1+p2)*0.5
                        if not point_in_poly(mid, [pts2d[i] for i in range(len(pts2d))]):
                            continue
                        # accept fallback diagonal
                        new_line = Line(int(g1), int(g2), LineType.FACET, torch.tensor(0.0))
                        local_added.append(new_line)
                        existing_edges.add(tuple(sorted((int(g1), int(g2)))))
                        # remove a vertex to reduce polygon size heuristically:
                        # remove the vertex between a_local and b_local (choose middle)
                        # here remove b_local (safe)
                        idxs.pop(b_local)
                        fallback_done = True
                        break
                    if fallback_done:
                        break

                if not fallback_done:
                    print("Warning: cannot find non-crossing ear — polygon may be non-simple or numerically degenerate")
                    break

        # at the end, idxs length should be 3; if >3 we failed; still accept local_added
        added_edges.extend(local_added)

    return added_edges

def triangulate_all(listPoints: list[Point], listLines: list[Line]) -> list[Line]:
    init_listLines = listLines.copy()
    # listLines_ = triangluate_poly(listPoints, listLines)
    
    polygons = find_polygons(listPoints, init_listLines)
    listLines_ = triangulate_polygons(listPoints, listLines, polygons)
    
    print("polygons:",len(polygons))
    # for poly in polygons:
    #     for i in range(len(poly)):
    #         print(poly[i])
    #     print("====")
    # for line in listLines_:
    #     print(line)
        
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
