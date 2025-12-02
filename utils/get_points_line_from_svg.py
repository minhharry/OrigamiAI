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

def point_is_in_line(point , A, B) -> bool:
    px, py = point
    ax, ay = A
    bx, by = B

    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > 1e-6: 
        return False

    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    if dot > 0: 
        return False

    return True


def check_intersection_line(p1 , p2, q1, q2) -> bool:
    
    EPS = 1e-6
    t_vec = q1 - q2
    s_vec = p2 - p1
    
    A = np.column_stack((t_vec, -s_vec))
    b = p1 - q2
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        a1 = 1 if point_is_in_line(p1, q1, q2) else 0
        a2 = 1 if point_is_in_line(p2, q1, q2) else 0
        a3 = 1 if point_is_in_line(q1, p1, p2) else 0
        a4 = 1 if point_is_in_line(q2, p1, p2) else 0
        if a1 + a2 == 2 or a3 + a4 == 2:
            return True
        else:
            return False
        
    s, t = np.linalg.solve(A, b)
    if 0-EPS <= s <= 1+EPS and 0-EPS <= t <= 1+EPS:
        a1 = 1 if point_is_in_line(p1, q1, q2) else 0
        a2 = 1 if point_is_in_line(p2, q1, q2) else 0
        a3 = 1 if point_is_in_line(q1, p1, p2) else 0
        a4 = 1 if point_is_in_line(q2, p1, p2) else 0
        if a1 + a2 == 2 or a3 + a4 == 2:
            return True
        elif a1 + a2 == 1:
            return False
        return True
    return False


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

def create_points_lines_v2(root) -> tuple[list[Point], list[Line]]:
    listPoints: list[Point] = []
    listLines: list[Line] = []
    has_border = False
    for child in root:
        if child.tag == LINE_TAG:
            if get_type_line_from_svg(child)==LineType.BORDER:
                has_border = True
    for child in root:
        if not has_border and child.tag == RECT_TAG:
            listPoints.append(Point(float(child.attrib['x']),0.0,float(child.attrib['y'])))
            listPoints.append(Point(float(child.attrib['x'])+float(child.attrib['width']),0.0,float(child.attrib['y'])))
            listPoints.append(Point(float(child.attrib['x'])+float(child.attrib['width']),0.0,float(child.attrib['y'])+float(child.attrib['height'])))
            listPoints.append(Point(float(child.attrib['x']),0.0,float(child.attrib['y'])+float(child.attrib['height'])))
            listLines.append(Line(0, 1, LineType.BORDER))
            listLines.append(Line(1, 2, LineType.BORDER))
            listLines.append(Line(2, 3, LineType.BORDER))
            listLines.append(Line(3, 0, LineType.BORDER))
            has_border = True
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
            if get_type_line_from_svg(child)==LineType.BORDER:
                has_border = True
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
    listPoints, listLines = create_points_lines_v2(root) #get all points , include intersection points
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

def triangulate_polygons(listPoints: list, listLines: list, polygons: list[list[int]]) -> list:
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

        idxs = list(range(m))
        local_added = []

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
                if ccw:
                    if cross <= EPS:
                        continue
                else:
                    if cross >= -EPS:
                        continue

                any_inside = False
                for j in idxs:
                    if j in (i_prev, i_curr, i_next):
                        continue
                    if point_in_triangle(pts2d[j], A, B, C):
                        any_inside = True
                        break
                if any_inside:
                    continue

                g1 = poly_indices[i_prev]
                g2 = poly_indices[i_next]
                if tuple(sorted((g1, g2))) in existing_edges:
                    pass
                
                p1 = pts2d[i_prev]; p2 = pts2d[i_next]
                intersects = False
                for t in range(len(idxs)):
                    j1 = idxs[t]; j2 = idxs[(t+1)%len(idxs)]
                    # if j1 in (i_prev, i_next) or j2 in (i_prev, i_next):
                    #     continue
                    q1 = pts2d[j1]; q2 = pts2d[j2]
                    if check_intersection_line(p1,p2,q1,q2):
                        intersects = True
                        break
                if intersects:
                    continue

                mid = (p1 + p2) * 0.5
                full_poly_xy = [pts2d[i] for i in range(len(pts2d))]
                if not point_in_poly(mid, full_poly_xy):
                    continue

                new_line = Line(int(g1), int(g2), LineType.FACET, torch.tensor(0.0))
                local_added.append(new_line)
                existing_edges.add(tuple(sorted((int(g1), int(g2)))))

                idxs.pop(k)
                ear_found = True
                break

            if not ear_found:
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
                            if check_intersection_line(p1,p2, pts2d[j1], pts2d[j2]):
                                intersects = True
                                break
                        if intersects:
                            continue
                        mid = (p1+p2)*0.5
                        if not point_in_poly(mid, [pts2d[i] for i in range(len(pts2d))]):
                            continue
                        new_line = Line(int(g1), int(g2), LineType.FACET, torch.tensor(0.0))
                        local_added.append(new_line)
                        existing_edges.add(tuple(sorted((int(g1), int(g2)))))
                        idxs.pop(b_local)
                        fallback_done = True
                        break
                    if fallback_done:
                        break

                if not fallback_done:
                    print("Warning: cannot find non-crossing ear — polygon may be non-simple or numerically degenerate")
                    break
        added_edges.extend(local_added)

    return added_edges

def triangulate_all(listPoints: list[Point], listLines: list[Line]) -> list[Line]:
    init_listLines = listLines.copy()
    
    polygons = find_polygons(listPoints, init_listLines)
    listLines_ = triangulate_polygons(listPoints, listLines, polygons)
        
    for line in listLines_:
        listLines.append(line)

    return listLines 




if __name__ == "__main__":
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    print("Points: ",len(listPoints))
    print("Lines: ",len(listLines))
