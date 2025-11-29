from ptu_board import Point, Line, visualize, point_is_in_line
import numpy as np


boundary_points = [
    Point(-1.0,-1.0,0,None),
    Point(1.0,-1.0,0,None),
    Point(4.0,0.0,0,None),
    Point(0,4.0,0,None),
    Point(-4.0,0.0,0,None),
    Point(-4.0,-5.0,0,None),
    Point(0,-5.0,0,None),
    Point(4.0,-5.0,0,None),
]

points = [
    Point(0,1,0,None),
    Point(0,-1,0,None),
    Point(0,0,0,None),
    Point(-1,-1,0,None),
    Point(0,-2,0,None),
    Point(1,-1,0,None),
    Point(4,0,0,None),
    Point(0,4,0,None),
    Point(-4,0,0,None),
    Point(-4,-5,0,None),
    Point(0,-5,0,None),
    Point(4,-5,0,None),
]

lines = [
    Line(points[0], points[2], 0),
    Line(points[1], points[2], 0),
    Line(points[1], points[3], 0),
    Line(points[1], points[4], 0),
    Line(points[1], points[5], 0),
    Line(points[0], points[6], 0),
    Line(points[0], points[7], 0),
    Line(points[0], points[8], 0),
    Line(points[4], points[9], 0),
    Line(points[4], points[10], 0),
    Line(points[4], points[11], 0),
]

def check_intersection_line(line1: Line, line2: Line) -> bool:
    p1 = line1.p1 #v_i
    p2 = line1.p2 #v_n
    q1 = line2.p1 #vq
    q2 = line2.p2 #vp

    EPS = 1e-6
    t_vec = q1.position[:2] - q2.position[:2]
    s_vec = p2.position[:2] - p1.position[:2]
    
    A = np.column_stack((t_vec, -s_vec))
    b = p1.position[:2] - q2.position[:2]
    det = np.linalg.det(A)

    if abs(det) < 1e-6:
        a1 = 1 if point_is_in_line(p1, (q1, q2)) else 0
        a2 = 1 if point_is_in_line(p2, (q1, q2)) else 0
        a1 = 1 if point_is_in_line(p1, (q1, q2)) else 0
        a2 = 1 if point_is_in_line(p2, (q1, q2)) else 0
        if a1 + a2 == 2:
            return True
        else:
            return False
        
    s, t = np.linalg.solve(A, b)
    if 0-EPS <= s <= 1+EPS and 0-EPS <= t <= 1+EPS:
        a1 = 1 if point_is_in_line(p1, (q1, q2)) else 0
        a2 = 1 if point_is_in_line(p2, (q1, q2)) else 0
        if a1 + a2 == 2:
            return True
        elif a1 + a2 == 1:
            return False
        return True
    return False

def check_intersection_all_line(line: Line, list_lines: list[Line]) -> bool:
    for i in range(len(list_lines)):
        if check_intersection_line(line, list_lines[i]):
            return True
    return False


def connect_boundary_points_recursive(points: list[Point], boundary_points: list[Point], list_lines: list[Line],order: list, start_idx: int, posible_idx: list[int]) -> list[tuple[Point, Point]]:
    if start_idx == len(boundary_points):
        return []
    temp_lines = list_lines.copy()
    idx = 0
    while idx < len(posible_idx):
        intersec = check_intersection_all_line(Line(boundary_points[order[start_idx]],boundary_points[order[posible_idx[idx]]],0),temp_lines)
        
        if intersec:
            idx += 1
            if idx == len(posible_idx):
                raise Exception("error intersection")
            continue
        try:
            temp_lines.append(Line(boundary_points[order[start_idx]],boundary_points[order[posible_idx[idx]]],0))
            picked = posible_idx[idx]
            posible_idx_new = posible_idx.copy()
            posible_idx_new.remove(picked)
        
            res = connect_boundary_points_recursive(points,boundary_points,temp_lines,order,picked,posible_idx_new)
            return [(boundary_points[order[start_idx]],boundary_points[order[posible_idx[idx]]])]+res
        except Exception as e:
            if idx == len(posible_idx)-1:
                raise e
            temp_lines.remove(temp_lines[len(temp_lines)-1])
            idx +=1
    return []

def connect_boundary_points(points: list[Point], boundary_points: list[Point], list_lines: list[Line]) -> list[tuple[Point, Point]]:
    if len(boundary_points) < 2:
        return []

    pts_2d = np.array([[p.position[0], p.position[1]] for p in boundary_points])
    centroid = np.mean(pts_2d, axis=0)

    angles:list[float] = np.arctan2(pts_2d[:, 1] - centroid[1], pts_2d[:, 0] - centroid[0])

    order= np.argsort(angles)
    boundary_lines = []
    try:
        boundary_lines = connect_boundary_points_recursive(points,boundary_points,lines,order,0,[x for x in range(1,len(order))]+[0])
    except Exception as e:
        print(e)

    return boundary_lines

if __name__ == "__main__":
    boundary_lines = connect_boundary_points(points,boundary_points,lines)
    visualize(points,[Line(p,q) for p,q in boundary_lines]+lines,True)