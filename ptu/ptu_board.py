from __future__ import annotations
import numpy as np
import random
import math
from enum import Enum
import json
import os
from datetime import datetime
NUM_IN = [1,0,0]
DEGREE = NUM_IN[0] + NUM_IN[1] + NUM_IN[2] + 3
EPS = 1e-5
BOARD_SIZE = 15
GRID_SIZE = BOARD_SIZE*2
ACTIONS = []

class Action(Enum):
    START = -1
    PICK_ROOT_POINT = 0
    PICK_NEW_POINT = 1
    MERGE_POINT = 2
    EXPAND_POINT = 3
    EXPAND_SYMMETRIC_X = 4
    EXPAND_SYMMETRIC_Y = 5
    END = 6
    END_STEP = 8
    DESTROY = 7

    def __dict__(self):
        return self.name
    def __str__(self):
        return f"Action({self.name})"
    def __repr__(self):
        return str(self)

class AutoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name  # OR obj.value
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)


class ActionData:
    def __init__(self, point: tuple[int,int], other_point = None) -> None:
        self.point = (int(point[0]),int(point[1]))
        self.other_point = other_point
        if other_point is not None:
            self.other_point = (int(other_point[0]),int(other_point[1]))

    def __str__(self) -> str:
        return f"ActionData({self.point}, {self.other_point})"

    def __repr__(self) -> str:
        return str(self)
    

class Strategy(Enum):
    RANDOM = 1
    BFS = 2
    DFS = 3
    EVO = 4
    PPO = 5

class SymmetricStrategy(Enum):
    X = 1
    Y = 2
    XY = 3
    X_AND_Y = 4
    No = 5

SYMMETRIC_DIR = SymmetricStrategy.Y

ptu_strategy = Strategy.BFS

def get_x_length():
    if SYMMETRIC_DIR == SymmetricStrategy.X:
        return BOARD_SIZE
    elif SYMMETRIC_DIR == SymmetricStrategy.Y:
        return BOARD_SIZE//2
    elif SYMMETRIC_DIR == SymmetricStrategy.XY:
        return BOARD_SIZE//2
    return BOARD_SIZE

def get_y_length():
    if SYMMETRIC_DIR == SymmetricStrategy.X:
        return BOARD_SIZE//2
    elif SYMMETRIC_DIR == SymmetricStrategy.Y:
        return BOARD_SIZE
    elif SYMMETRIC_DIR == SymmetricStrategy.XY:
        return BOARD_SIZE//2
    return BOARD_SIZE

class Point:
    def __init__(self, x: float, y: float, z: float, point_root: Point = None):
        self.position = np.array([x,y,z])
        self.in_diheral_angles = []
        self.out_diheral_angles = []
        self.point_root = [point_root]
    
    def __str__(self):
        return f"Point({self.position[0]},{self.position[1]},{self.position[2]})"

    def __eq__(self, other):
        if not isinstance(other, Point): return False
        return np.allclose(self.position, other.position, atol=1e-6)

    def __hash__(self):
        x,y,z = self.position
        return hash((round(float(x),6), round(float(y),6), round(float(z),6)))

class Line:
    def __init__(self, p1: Point, p2: Point, targetTheta: float = 0):
        self.p1 = p1
        self.p2 = p2
        self.targetTheta = targetTheta

def is_on_symmetric(p1: Point)->bool:
    if SYMMETRIC_DIR == SymmetricStrategy.X:
        return p1.position[1] ==  0
    elif SYMMETRIC_DIR == SymmetricStrategy.Y:
        return p1.position[0] == 0
    elif SYMMETRIC_DIR == SymmetricStrategy.X_AND_Y:
        return p1.position[0] == 0 or p1.position[1] == 0
    return False

def calc_angles(root_point: Point, p1: Point, p2: Point, p3: Point) -> tuple[list[list[float]], list[list[float]], Point, Point, Point]:
    # print("calc angles",root_point)
    points = root_point.point_root + [p1, p2, p3]
    # for i in points:
    #     print(i)
    vecs = [p.position - root_point.position for p in points]
    angles = [np.arctan2(v[1], v[0]) for v in vecs]
    
    in_angles = root_point.in_diheral_angles + [-999, -999, -999]
    # print("in_angles:",len(in_angles))
    # print("angles:",len(angles))
    # print("points:",len(points))
    sorted_pairs = sorted(zip(points, in_angles, angles), key=lambda x: x[2])
    sorted_points, sorted_in_angles, sorted_angles = zip(*sorted_pairs)
    n = len(sorted_points)
    # print(n)
    p1_new = None
    p2_new = None
    p3_new = None
    i = 0
    while p1_new is None or p2_new is None or p3_new is None:
        if sorted_points[i] in [p1,p2,p3]:
            if p3_new is not None:
                if sorted_points[i] in [p1,p2,p3]:
                    p1_new = sorted_points[i]
                    p2_new = [p for p in [p1,p2,p3] if p not in [p1_new, p3_new]][0]
            else:
                next_idx = (i + 1) % n
                if sorted_points[next_idx] not in [p1,p2,p3]:
                    p3_new = sorted_points[i]
        i = (i + 1) % n
    # print("====")
    # print("p1_new:",p1_new)
    # print("p2_new:",p2_new)
    # print("p3_new:",p3_new)
    # for i in range(len(sorted_points)):
    #     print(i,sorted_points[i])
    i1 = sorted_points.index(p1_new)
    i2 = sorted_points.index(p2_new)
    i3 = sorted_points.index(p3_new)
    # print("ok")

    def sector_angles_ccw(start_idx: int, end_idx: int, not_include: int):
        angles_seg = []
        in_seg = []
        idx = start_idx
        step = 1
        try:
            while True:
                if idx == not_include:
                    angles_seg = []
                    in_seg = []
                    idx = start_idx
                    step = -1
                    continue
                next_idx = (idx + step) % n
                u = sorted_points[idx].position - root_point.position
                v = sorted_points[next_idx].position - root_point.position
                angle_af = np.arctan2(v[1], v[0])
                angle_bf = np.arctan2(u[1], u[0])
                
                if angle_af < 0:
                    angle_af = angle_af + np.pi*2
                if angle_bf < 0:
                    angle_bf = angle_bf + np.pi*2

                angle = angle_af - angle_bf
                if angle < 0:
                    angle = angle + np.pi*2
                angles_seg.append(angle)
                if sorted_in_angles[idx] != -999:
                    in_seg.append(sorted_in_angles[idx])
                
                if next_idx == end_idx:
                    break
                idx = next_idx
        except:
            print("Error: sector_angles_ccw")
        return angles_seg, in_seg
    
    seg1_d, seg1_in = sector_angles_ccw(i3, i1, i2)
    seg2_d, seg2_in = sector_angles_ccw(i1, i2, i3)
    seg3_d, seg3_in = sector_angles_ccw(i2, i3, i1)
    
    return ([seg1_d, seg2_d, seg3_d],
            [seg1_in, seg2_in, seg3_in],
            p1_new, p2_new, p3_new)

def Rx(theta):
  return np.asarray([[ 1, 0           , 0     ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])

def Rz(theta):
  return np.asarray([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0         , 0        , 1 ]])

def calc_p_j_m(p0,sector_angles,list_in_diheral_angles):
    res = np.identity(3)
    m = len(sector_angles)
    for i in range(m-1):
        res = np.matmul(res,transform_fold_forw(sector_angles[i], list_in_diheral_angles[i]))
    res = res.dot(Rz(sector_angles[-1]))
    p_m = np.matmul(res,p0)
    return p_m

def transform_fold_forw(angle_z,angle_x):
    return np.matmul(Rz(angle_z),Rx(angle_x))

def transform_fold_rev(angle_x,angle_z):
    return np.matmul(Rx(angle_x),Rz(angle_z))

def beta_delta(p1, pm, u_j, alpha_j_1):
    gamma = np.arccos(np.clip(p1.dot(pm), -1., 1.))
    if np.isclose(pm[2],2., rtol = 1e-05, atol = 1e-8, equal_nan=False):
        pm_z = 1e-05
    else:
        pm_z = pm[2]
    sgn = -np.sign(pm_z)
    numerator = np.cos(gamma) - np.cos(alpha_j_1)*np.cos(u_j)
    denominator = np.sin(alpha_j_1)*np.sin(u_j)
    if abs(denominator) < EPS:
        denominator = np.sign(numerator)*EPS
    
    temp = np.clip(numerator/denominator,-1.,1.)
    return sgn*np.arccos(temp)

def calculate_theta(u1 ,u2 ,u3):
    def theta(u1,u2,u3):
       numerator = np.cos(u1) - np.cos(u2)*np.cos(u3)
       denominator = np.sin(u2)*np.sin(u3)
       argcos = np.clip(numerator/denominator,-1.,1)
       return np.acos(argcos)

    theta1 = theta(u1,u2,u3)
    theta2 = theta(u2,u1,u3)
    theta3 = theta(u3,u1,u2)
    return theta1, theta2, theta3
    
def compute_folded_unit(alpha_arr,rho):
   p_0 = np.array([1,0,0])
   res = np.identity(3)    

   for i in range(len(alpha_arr)-1):
        res = np.matmul(res,transform_fold_forw(alpha_arr[i], rho[i]))
   
   res = res.dot(Rz(alpha_arr[-1]))
   p_m = np.matmul(res,p_0)

   return np.arccos(np.clip(p_0.dot(p_m), -1., 1.))

def calc(sector_angles:list[list[float]],list_in_diheral_angles:list[list[float]]) -> tuple[list[float],list[float]]:
    
    def calc_beta_delta(index):
        p0 = np.array([1,0,0])
        res = np.identity(3)    

        m = len(sector_angles[index])
        if m > 1:
            for i in range(len(sector_angles[index])-1):
                res = np.matmul(res,transform_fold_forw(sector_angles[index][i], list_in_diheral_angles[index][i]))
            res = res.dot(Rz(sector_angles[index][-1]))
            p_m = np.matmul(res,p0)
            p_1 = np.dot(Rz(sector_angles[index][0]),p0)
            u = compute_folded_unit(sector_angles[index],list_in_diheral_angles[index])
            beta = beta_delta(p_1,p_m,u,sector_angles[index][0])

            temp_sector_angles = sector_angles[index].copy()
            temp_sector_angles.reverse()
            temp_list_in_diheral_angles = list_in_diheral_angles[index].copy()
            temp_list_in_diheral_angles.reverse()
            p_j_0_revert = calc_p_j_m(p0,temp_sector_angles,temp_list_in_diheral_angles)
            p_jm1_revert = np.dot(Rz(temp_sector_angles[0]),p0)
            u = compute_folded_unit(temp_sector_angles,temp_list_in_diheral_angles)
            delta = beta_delta(p_jm1_revert, p_j_0_revert,u,temp_sector_angles[0])
            # m_revert = np.identity(3)    
            # m_revert = np.matmul(Rz(-sector_angles[index][-1]),m_revert)
            # alpha_vec_revert = sector_angles[index].copy()
            # alpha_vec_revert = alpha_vec_revert[:-1]
            # list_in_diheral_angles_revert = list_in_diheral_angles[index].copy()
            # list_in_diheral_angles_revert.reverse()
            # for i in range(m-1):
            #     m_revert = np.matmul(m_revert,transform_fold_rev(-list_in_diheral_angles_revert[i], -alpha_vec_revert[i]))

            # p_j_0_revert = np.matmul(m_revert,p0)
            # if abs(p_j_0_revert[2]) < 1e-3:
            #     p_j_0_revert[2] = 0
            # print("p_j_0_revert:",p_j_0_revert)
            # p_jm1_revert = np.matmul(Rz(-sector_angles[index][-1]),p0)
            # print("p_jm1_revert:",p_jm1_revert)
            # delta = beta_delta(p_jm1_revert, p_j_0_revert,u,sector_angles[index][0])
            return beta,delta
        else:
            m_revert = Rz(sector_angles[index][-1])
            u = sector_angles[index][0]
            beta = 0
            delta = 0
            return beta,delta
    u1 = compute_folded_unit(sector_angles[0],list_in_diheral_angles[0])
    u2 = compute_folded_unit(sector_angles[1],list_in_diheral_angles[1])
    u3 = compute_folded_unit(sector_angles[2],list_in_diheral_angles[2])

    arr_u = np.array([u1,u2,u3])
    arr_u.sort()
    
    beta1,delta1 = calc_beta_delta(0)
    beta2,delta2 = calc_beta_delta(1)
    beta3,delta3 = calc_beta_delta(2)

    theta1, theta2, theta3 = calculate_theta(u1,u2,u3)

    phi1 = beta3 + np.pi - theta1 + delta2
    phi2 = beta1 + np.pi - theta2 + delta3
    phi3 = beta2 + np.pi - theta3 + delta1

    if phi1 > np.pi:
        phi1 -= 2*np.pi
    elif phi1 < -np.pi:
        phi1 += 2*np.pi

    if phi2 > np.pi:
        phi2 -= 2*np.pi
    elif phi2 < -np.pi:
        phi2 += 2*np.pi
    
    if phi3 > np.pi:
        phi3 -= 2*np.pi
    elif phi3 < -np.pi:
        phi3 += 2*np.pi

    M1:list[float] = [phi1,phi2,phi3]

    phi1 = beta3 + theta1 - np.pi + delta2
    phi2 = beta1 + theta2 - np.pi + delta3
    phi3 = beta2 + theta3 - np.pi + delta1

    if phi1 > np.pi:
        phi1 -= 2*np.pi
    elif phi1 < -np.pi:
        phi1 += 2*np.pi

    if phi2 > np.pi:
        phi2 -= 2*np.pi
    elif phi2 < -np.pi:
        phi2 += 2*np.pi
    
    if phi3 > np.pi:
        phi3 -= 2*np.pi
    elif phi3 < -np.pi:
        phi3 += 2*np.pi

    if (arr_u[0]+arr_u[1] - arr_u[2] > -0.01):
        pass
    else:
        print("KHONG GAP DUOC DAU HEHE ====================")
        return [],[]
    
    M2:list[float] = [phi1,phi2,phi3]
    return M1,M2

def calc_ptu(sector_angles:list[list[float]],list_in_diheral_angles:list[list[float]]) -> list[list[float]]:
   if (not sector_angles or not list_in_diheral_angles):
       print(len(sector_angles),len(list_in_diheral_angles))
       print("Error: sector_angles or list_in_diheral_angles is empty")
       return [[],[],[]]

   M1, M2 = calc(sector_angles,list_in_diheral_angles)

   if M1 == [] or M2 == []:
       print("Error: M1 or M2 is empty")
       return [[],[],[]]
   M1 = [float(M1[2]),float(M1[0]),float(M1[1])]
   M2 = [float(M2[2]),float(M2[0]),float(M2[1])]
   t = [x for y in sector_angles for x in y]
   return [t, M1,M2]

def pick_up_point(boundary_points: list[Point]):
    assert len(boundary_points) > 0, "error: boundary_points is empty!"
    idx = random.randrange(0,len(boundary_points))
    ACTIONS.append(
    {
        "action": Action.PICK_ROOT_POINT,
        "action_data": ActionData((boundary_points[idx].position[0],boundary_points[idx].position[1]))
    })
    return idx

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
        a3 = 1 if point_is_in_line(q1, (p1, p2)) else 0
        a4 = 1 if point_is_in_line(q2, (p1, p2)) else 0
        if a1 + a2 == 2 or a3 + a4 == 2:
            return True
        else:
            return False
        
    s, t = np.linalg.solve(A, b)
    print("s,t:",s,t)
    if 0-EPS <= s <= 1+EPS and 0-EPS <= t <= 1+EPS:
        a1 = 1 if point_is_in_line(p1, (q1, q2)) else 0
        a2 = 1 if point_is_in_line(p2, (q1, q2)) else 0
        a3 = 1 if point_is_in_line(q1, (p1, p2)) else 0
        a4 = 1 if point_is_in_line(q2, (p1, p2)) else 0
        print("a1,a2,a3,a4:",a1,a2,a3,a4)
        if a1 + a2 == 2 or a3 + a4 == 2:
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
                raise Exception()
            continue
        try:
            temp_lines.append(Line(boundary_points[order[start_idx]],boundary_points[order[posible_idx[idx]]],0))
            polygons = find_polygons(temp_lines)
            print("temp_lines:",len(temp_lines))
            for i in temp_lines:
                print(i.p1,i.p2)
            print("polygons:",len(polygons))
            for i in polygons:
                print("===")
                for j in i:
                    print(j.p1,j.p2)
            for i in boundary_points:
                if is_in_polygons(i,polygons):
                    idx += 1
                    if idx == len(posible_idx):
                        raise Exception()
                    continue

            visualize(points,temp_lines,True)
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
        boundary_lines = connect_boundary_points_recursive(points,boundary_points,list_lines,order,0,[x for x in range(1,len(order))]+[0])
    except Exception as e:
        print(e)

    return boundary_lines

def point_is_in_line(point: Point, line: tuple[Point,Point]) -> bool:
    A, B = line
    px, py,_ = point.position
    ax, ay,_ = A.position
    bx, by,_ = B.position

    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > 1e-6: 
        return False

    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    if dot > 0: 
        return False

    return True

def get_point_with_position(points: list[Point], position: np.ndarray) -> Point:
    for point in points:
        if point.position[0] == position[0] and point.position[1] == position[1] and point.position[2] == position[2]:
            return point
    return None

def check_intersection(list_points: list[Point], list_lines: list[tuple[Point,Point]], new_points: list[Point], v_i: Point ) -> bool:

# vi - vp = -s(vN+j − vi) + t(vq − vp)
    if len(list_points) <= 3:
        return False
    list_lines_ :list[tuple[Point,Point]] = list_lines.copy()
    for i in range(len(new_points)):
        v_i_2 = v_i.position[:2]
        v_N_j = new_points[i].position[:2]
        s_vec = v_N_j - v_i_2 # vNj - vi         
        for j in range(len(list_lines_)):
            if list_lines_[j][0] == new_points[i] or list_lines_[j][1] == new_points[i]:
                continue
            v_p = list_lines_[j][0].position[:2] 
            v_q = list_lines_[j][1].position[:2]
            t_vec = v_q - v_p
            A = np.column_stack((t_vec, -s_vec))
            b = v_i_2 - v_p
            det = np.linalg.det(A)

            if abs(det) < EPS:
                a1 = 1 if point_is_in_line(v_i, (list_lines_[j][0], list_lines_[j][1])) else 0
                a2 = 1 if point_is_in_line(new_points[i], (list_lines_[j][1], list_lines_[j][0])) else 0
                a3 = 1 if point_is_in_line(list_lines_[j][0], (v_i, new_points[i])) else 0
                a4 = 1 if point_is_in_line(list_lines_[j][1], (v_i, new_points[i])) else 0
                if a1 + a2 == 2 or a3 + a4  == 2:
                    return True
                elif a1 + a2 + a3 + a4 == 1:
                    x1, y1 = position_revert(v_i.position)
                    x2, y2 = position_revert(new_points[i].position)
                    x3, y3 = position_revert(list_lines_[j][0].position)
                    x4, y4 = position_revert(list_lines_[j][1].position)

                    if [x1, y1] == [x2, y2] or [x1, y1] == [x3, y3] or [x1, y1] == [x4, y4]:
                        continue
                    elif [x2, y2] == [x3, y3] or [x2, y2] == [x4, y4]:
                        continue
                    elif [x3, y3] == [x4, y4]:
                        continue
                else:
                    continue

            s, t = np.linalg.solve(A, b)
            if 0-EPS <= s <= 1+EPS and 0-EPS <= t <= 1+EPS:
               
                a1 = 1 if point_is_in_line(v_i, (list_lines_[j][0], list_lines_[j][1])) else 0
                a2 = 1 if point_is_in_line(new_points[i], (list_lines_[j][1], list_lines_[j][0])) else 0
                a3 = 1 if point_is_in_line(list_lines_[j][0], (v_i, new_points[i])) else 0
                a4 = 1 if point_is_in_line(list_lines_[j][1], (v_i, new_points[i])) else 0
                if a1 + a2 == 2 or  a3 + a4 == 2:
                    return True
                elif a1 + a2 + a3 + a4 == 1:
                    x1, y1 = position_revert(v_i.position)
                    x2, y2 = position_revert(new_points[i].position)
                    x3, y3 = position_revert(list_lines_[j][0].position)
                    x4, y4 = position_revert(list_lines_[j][1].position)
                    if [x1, y1] == [x2, y2] or [x1, y1] == [x3, y3] or [x1, y1] == [x4, y4]:
                        continue
                    elif [x2, y2] == [x3, y3] or [x2, y2] == [x4, y4]:
                        continue
                    elif [x3, y3] == [x4, y4]:
                        continue
                    return True
                intersec = v_p + s * t_vec
                # check distance to all 4 endpoints
                if (
                    np.linalg.norm(intersec - v_p) < EPS*100 or
                    np.linalg.norm(intersec - v_q) < EPS*100 or
                    np.linalg.norm(intersec - v_i_2) < EPS*100 or
                    np.linalg.norm(intersec - v_N_j) < EPS*100
                ):
                    u1 = v_i_2 - v_N_j
                    u2 = v_p - v_q
                    
                    alpha = np.acos(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))
                    if abs(alpha) < 0.01 or abs(alpha - np.pi) < 0.01:
                        return True 
                    else:
                        continue
                return True
                            

    return False

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1-p2))

def get_center_point(points: list[Point]) -> Point:
    positions = np.array([p.position for p in points])
    
    # tính trung bình theo từng trục
    center = positions.mean(axis=0)

    # Tạo point mới
    return Point(center[0], center[1], center[2], point_root=None)

def polygon_to_points(polygon: list[Line]) -> list[Point]:
    """Chuyển list[Line] → list[Point] theo đúng thứ tự."""
    points = [polygon[0].p1]
    for line in polygon:
        if line.p1 not in points:
            points.append(line.p1)
        if line.p2 not in points:
            points.append(line.p2)
    centerPoint = get_center_point(points)
    points, _ = sort_points_ccw(points, list(np.zeros(len(points))), centerPoint)
    return points

def is_on_segment(point: Point, segment: Line) -> bool:
    p = point.position
    a = segment.p1.position
    b = segment.p2.position
   
    if np.allclose(a, b):
        return np.allclose(p, a)

    vec_ap = p - a
    vec_ab = b - a
    
    cross_product_mag = np.linalg.norm(np.cross(vec_ap, vec_ab))
    
    is_collinear = np.isclose(cross_product_mag, 0)
    
    if not is_collinear:
        return False
    
    vec_bp = p - b
    dot_product = np.dot(vec_ap, vec_bp)
    
    is_between = (dot_product <= 0)

    return is_between

def is_point_in_polygon_2d(pt: Point, points: list[Point], lines: list[Line]) -> bool:
    x, y = pt.position[0], pt.position[1]
    inside = False
    n = len(points)
    for i in range(len(lines)):
        if is_on_segment(pt, lines[i]):
            return False
    for i in range(n):
        x1, y1 = points[i].position[0], points[i].position[1]
        x2, y2 = points[(i + 1) % n].position[0], points[(i + 1) % n].position[1]

        # Kiểm tra xem cạnh có giao với tia ngang đi qua pt hay không
        intersects = ((y1 > y) != (y2 > y)) and \
                     (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)

        if intersects:
            inside = not inside

    return inside


def is_in_polygons(consider_point: Point,  polygons: list[list[Line]]) -> bool:
    for polygon in polygons:
        points = polygon_to_points(polygon)
        if is_point_in_polygon_2d(consider_point, points,polygon):
            return True
    return False
     
def sort_points_ccw(points: list[Point], data: list, center: Point):
    paired = list(zip(points, data))
    
    def angle(pair):
        p = pair[0]
        dx = p.position[0] - center.position[0]
        dy = p.position[1] - center.position[1]
        return math.atan2(dy, dx)
    
    paired_sorted = sorted(paired, key=angle)
    sorted_points, sorted_data = zip(*paired_sorted)    
    return list(sorted_points), list(sorted_data)
#chua xet lai base point sau khi merge

def get_angle_2d(p_center: Point, p_neighbor: Point) -> float:
    """
    Tính góc 2D (trên mặt phẳng XY) của vector từ p_center đến p_neighbor.
    Góc được tính theo chuẩn atan2 (ngược chiều kim đồng hồ, 0 độ ở +X).
    """
    pos_c = p_center.position
    pos_n = p_neighbor.position
    # Bỏ qua Z (pos[2])
    return math.atan2(pos_n[1] - pos_c[1], pos_n[0] - pos_c[0])

# =====================================================================
# HÀM TÌM KIẾM POLYGON
# =====================================================================
from collections import defaultdict
def find_polygons(lines: list[Line]) -> list[list[Line]]:
    adj_list = defaultdict(list)
    edge_map = {}

    for line in lines:
        p1, p2 = line.p1, line.p2
        
        adj_list[p1].append(p2)
        adj_list[p2].append(p1)
        
        # Lưu cả hai hướng để tra cứu Line object
        edge_map[(p1, p2)] = line
        edge_map[(p2, p1)] = line

    # 2. Sắp xếp danh sách kề theo góc (ngược chiều kim đồng hồ - CCW)
    sorted_adj_list = {}
    for point, neighbors in adj_list.items():
        # Sắp xếp các hàng xóm dựa trên góc 2D
        sorted_neighbors = sorted(neighbors, key=lambda n: get_angle_2d(point, n))
        sorted_adj_list[point] = sorted_neighbors

    # 3. Duyệt tìm các mặt (polygons)
    # visited_directed_edges: set((Point, Point))
    # Dùng để theo dõi các cạnh *có hướng* đã đi qua
    visited_directed_edges = set() 
    polygons = []

    for start_line in lines:
        # Mỗi cạnh phải được kiểm tra theo cả hai hướng, vì nó
        # là một phần của hai đa giác khác nhau (trừ khi ở biên)
        directions = [(start_line.p1, start_line.p2), (start_line.p2, start_line.p1)]
        
        for start_p1, start_p2 in directions:
            
            # Nếu đã đi theo hướng này rồi (là 1 phần của đa giác khác)
            if (start_p1, start_p2) in visited_directed_edges:
                continue 

            # Bắt đầu duyệt một đa giác mới
            current_polygon_lines = []
            
            # Bắt đầu đi từ start_p1 -> start_p2
            prev_point = start_p1
            current_point = start_p2
            
            while True:
                # Đánh dấu cạnh (có hướng) là đã duyệt
                visited_directed_edges.add((prev_point, current_point))
                
                # Lấy Line object và thêm vào đa giác hiện tại
                current_line = edge_map[(prev_point, current_point)]
                if current_line not in current_polygon_lines:
                    current_polygon_lines.append(current_line)
                
                # Lấy danh sách hàng xóm ĐÃ SẮP XẾP của điểm hiện tại
                sorted_neighbors = sorted_adj_list.get(current_point)
                
                if not sorted_neighbors:
                    break 

                try:
                    prev_index = sorted_neighbors.index(prev_point)
                except ValueError:
                    break 

                next_point = sorted_neighbors[(prev_index + 1) % len(sorted_neighbors)]
                
                prev_point = current_point
                current_point = next_point

                if prev_point == start_p1 and current_point == start_p2:
                    polygons.append(current_polygon_lines)
                    break
                
                if len(current_polygon_lines) > len(lines) + 1:
                    break 
                    
    return polygons

def is_valid_to_merge(points: list[Point], boundary_points: list[Point],exist_points: Point, v:Point ,lines: list[Line]) -> bool:
    temp_lines = lines.copy()
    if exist_points not in boundary_points:
        return False
    print("is valid to merge",v)
    print("exist_points:",exist_points)
    temp_lines = [ x for x in temp_lines if x.p1 is not v and x.p2 is not v]
    temp_lines.append(Line(v.point_root[0],exist_points,-999))
    
    polygons = find_polygons(lines)
    for i in boundary_points:
        if is_in_polygons(i,polygons):
            print("Error: is_in_polygon:",i)
            return False
    return True

def merge_points(points: list[Point], lines: list[Line], boundary_points: list[Point], v:Point, list_merge_points: list[Point]):
    print("merge points",v)
    print("list_merge_points:",list_merge_points[0])
    ACTIONS.append( {"action": Action.MERGE_POINT, "action_data": ActionData((v.position[0],v.position[1]), (list_merge_points[0].position[0],list_merge_points[0].position[1]))})
    merge_point = v
    for i in list_merge_points:
        if i in boundary_points:
            boundary_points.remove(i)
    
    for i in list_merge_points[::-1]:
        if i in points:
            points.remove(i)
    # visualize(points,lines,True)
    for i in range(len(lines)):
        if lines[i].p1 in list_merge_points:
            lines[i] = Line(merge_point,lines[i].p2,lines[i].targetTheta)
        elif lines[i].p2 in list_merge_points:
            lines[i] = Line(lines[i].p1,merge_point,lines[i].targetTheta)

# # #TODO: chua kiem tra
    merge_root_point = merge_point.point_root
    list_merge_root_point = merge_root_point
    for i in list_merge_points[0].point_root:
        if i not in list_merge_root_point:
            list_merge_root_point.append(i)
    list_driving_angles =  [y for x in list_merge_root_point for y in x.in_diheral_angles ]
    sorted_point, sorted_driving_angle = sort_points_ccw(list_merge_root_point,list_driving_angles, merge_point)

    merge_point.point_root = sorted_point
    merge_point.in_diheral_angles =  sorted_driving_angle
    
def create_points(root_point: Point, points: list[Point], sector_angles: list[list[float]], list_diheral_angles: list[float]) -> list[Point]:

    root_point = root_point
    print("create points",root_point)
    base_point = root_point.point_root[0]
    p_tuong_doi = base_point.position - root_point.position
    p1 = calc_p_j_m(p_tuong_doi,sector_angles[0][1:],list_diheral_angles)
    p2 = calc_p_j_m(p1,sector_angles[1],list_diheral_angles[len(sector_angles[0]):])
    p3 = calc_p_j_m(p2,sector_angles[2],list_diheral_angles[len(sector_angles[0])+len(sector_angles[1]):])

    p1 = p1 * random.randrange(1,3)/2.0
    p2 = p2 * random.randrange(1,3)/2.0
    p3 = p3 * random.randrange(1,3)/2.0

    p1 = p1 + root_point.position
    p2 = p2 + root_point.position   
    p3 = p3 + root_point.position

    point1 = Point(p1[0],p1[1],p1[2],root_point)
    point2 = Point(p2[0],p2[1],p2[2],root_point)
    point3 = Point(p3[0],p3[1],p3[2],root_point)
    point1.in_diheral_angles = [list_diheral_angles[len(sector_angles[0])-1]]
    point2.in_diheral_angles = [list_diheral_angles[len(sector_angles[0])+len(sector_angles[1])-1]]
    point3.in_diheral_angles = [list_diheral_angles[len(sector_angles[0])+len(sector_angles[1])+len(sector_angles[2])-1]]

    return [point1,point2,point3]

def pick_position(root_point: Point) -> list[tuple[int,int]]:
    # print("PICK POS")
    if SYMMETRIC_DIR == SymmetricStrategy.X:
        p1 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
        point_1 = Point(p1[0],p1[1],0,None)
        if is_on_symmetric(root_point) and not is_on_symmetric(point_1):
            p2 = (point_1.position[0],-point_1.position[1])
            p3 = ((random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),0)
        elif is_on_symmetric(root_point):
            p2 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
            p3 = (p2[0],-p2[1])
        else:
            p2 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
            p3 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
    elif SYMMETRIC_DIR == SymmetricStrategy.Y:
        p1 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
        point_1 = Point(p1[0],p1[1],0,None)
        if is_on_symmetric(root_point) and not is_on_symmetric(point_1):
            p2 = (-point_1.position[0],point_1.position[1])
            p3 = (0,(random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2))
        elif is_on_symmetric(root_point):
            p2 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
            p3 = (-p2[0],p2[1])
        else:
            p2 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
            p3 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
    elif SYMMETRIC_DIR == SymmetricStrategy.X_AND_Y:
        p1 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
        point_1 = Point(p1[0],p1[1],0,None)
        if is_on_symmetric(root_point) and not is_on_symmetric(point_1):
            if root_point.position[0] == 0:
                p2 = (-point_1.position[0],point_1.position[1])
                p3 = (0,(random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2))
            elif root_point.position[1] == 0:
                p2 = (point_1.position[0],-point_1.position[1])
                p3 = ((random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),0)
        else:
            p2 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
            p3 = (random.randrange(0,BOARD_SIZE//2+1)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE//2+1))
    else:
        p1 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
        p2 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
        p3 = (random.randrange(0,BOARD_SIZE)-BOARD_SIZE//2),(BOARD_SIZE//2 - random.randrange(0,BOARD_SIZE))
    ACTIONS.append(
    {
        "action": Action.PICK_NEW_POINT, 
        "action_data": ActionData(p1)
    })
    ACTIONS.append(
    {
        "action": Action.PICK_NEW_POINT,
        "action_data": ActionData(p2)
    })
    ACTIONS.append(
    {
        "action": Action.PICK_NEW_POINT,
        "action_data": ActionData(p3)
    })
    return [p1,p2,p3]

def is_valid(root_point: Point, p1: Point, p2: Point, p3: Point,  points: list[Point], boundary_points: list[Point], lines: list[Line],board: np.ndarray) -> bool:
    if  check_intersection(points, [(x.p1, x.p2) for x in lines],
                               [p1,p2,p3], root_point):
        # print("Intersec")
        return False
    polygons = find_polygons(lines)
    if is_in_polygons(p1,polygons):
        return False
    if is_in_polygons(p2,polygons):
        return False
    if is_in_polygons(p3,polygons):
        return False
    if p1 not in boundary_points:
        x, y = position_revert(p1.position)
        if board[y][x] != 0:
            return False
    if p2 not in boundary_points:
        x, y = position_revert(p2.position)
        if board[y][x] != 0:
            return False
    if p3 not in boundary_points:
        x, y = position_revert(p3.position)
        if board[y][x] != 0:
            return False
    return True

def create_base(points: list[Point], lines: list[Line], boundary_points: list[Point], driving_angles:float, board: np.ndarray) -> tuple[list[Point],list[Point],list[Line]]:
    # p1 = Point(0,1,0,None)
    # p2 = Point(1,0,0,None)
    # p3 = Point(0,-1,0,None)
    # p4 = Point(-1,0,0,None)

    p5 = Point(1,1,0,None)
    p6 = Point(1,-1,0,None)
    p7 = Point(-1,1,0,None)
    p8 = Point(-1,-1,0,None)

    # points.append(p1)
    # points.append(p2)
    # points.append(p3)
    # points.append(p4)
    points.append(p5)
    points.append(p6)
    points.append(p7)
    points.append(p8)

    # p5.point_root = [p2,p1]
    # p6.point_root = [p2,p3]
    # p7.point_root = [p4,p1]
    # p8.point_root = [p3,p4]

    p5.point_root = [p6,p7]
    p6.point_root = [p8,p5]
    p7.point_root = [p5,p8]
    p8.point_root = [p7,p6]

    p5.in_diheral_angles = [driving_angles,driving_angles]
    p6.in_diheral_angles = [driving_angles,driving_angles]
    p7.in_diheral_angles = [driving_angles,driving_angles]
    p8.in_diheral_angles = [driving_angles,driving_angles]

    # lines.append(Line(p1,p5,driving_angles))
    # lines.append(Line(p1,p7,driving_angles))
    # lines.append(Line(p2,p5,driving_angles))
    # lines.append(Line(p2,p6,driving_angles))
    # lines.append(Line(p3,p6,driving_angles))
    # lines.append(Line(p3,p8,driving_angles))
    # lines.append(Line(p4,p7,driving_angles))
    # lines.append(Line(p4,p8,driving_angles))

    lines.append(Line(p6,p5,driving_angles))
    lines.append(Line(p5,p7,driving_angles))
    lines.append(Line(p6,p8,driving_angles))
    lines.append(Line(p7,p8,driving_angles))

    # x,y = position_revert(p1.position)
    # print("OK")
    # board[y][x] = 1
    # x,y = position_revert(p2.position)
    # board[y][x] = 1
    # x,y = position_revert(p3.position)
    # board[y][x] = 1
    # x,y = position_revert(p4.position)
    # board[y][x] = 1
    x,y = position_revert(p5.position)
    board[y][x] = 1
    x,y = position_revert(p6.position)
    board[y][x] = 1
    x,y = position_revert(p7.position)
    board[y][x] = 1
    x,y = position_revert(p8.position)  
    board[y][x] = 1
    print(board)

    # boundary_points.append(p1)
    # boundary_points.append(p2)
    # boundary_points.append(p3)
    # boundary_points.append(p4)
    boundary_points.append(p5)
    boundary_points.append(p6)
    boundary_points.append(p7)
    boundary_points.append(p8)
    # boundary_points.append(p6)
    # boundary_points.append(p7)  
    # boundary_points.append(p8)
    return points, boundary_points, lines

def create_base_Y(points: list[Point], lines: list[Line], boundary_points: list[Point], driving_angles:float, board: np.ndarray) -> tuple[list[Point],list[Point],list[Line]]:
    p1 = Point(0,1,0,None)
    p2 = Point(0,-1,0,None)
    p3 = Point(0,0,0,None)

    points.append(p1)
    points.append(p2)
    points.append(p3)

    p1.in_diheral_angles = [driving_angles]
    p2.in_diheral_angles = [driving_angles]

    p1.point_root = [p3]
    p2.point_root = [p3]

    lines.append(Line(p1,p3,driving_angles))
    lines.append(Line(p2,p3,driving_angles))

    x,y = position_revert(p1.position)
    board[y][x] = 1
    x,y = position_revert(p2.position)
    board[y][x] = 1
    x,y = position_revert(p3.position)
    board[y][x] = 1
    
    boundary_points.append(p1)
    boundary_points.append(p2)

    return points, boundary_points, lines


def create_base_X_and_Y(points: list[Point], lines: list[Line], boundary_points: list[Point], driving_angles:float, board: np.ndarray) -> tuple[list[Point],list[Point],list[Line]]:
    p1 = Point(0,0,0,None)
    
    p2 = Point(0,1,0,p1)
    p3 = Point(1,0,0,p1)
    p4 = Point(0,-1,0,p1)
    p5 = Point(-1,0,0,p1)

    points.append(p1)
    points.append(p2)
    points.append(p3)
    points.append(p4)
    points.append(p5)

    p2.in_diheral_angles = [driving_angles]
    p3.in_diheral_angles = [-driving_angles]
    p4.in_diheral_angles = [-driving_angles]
    p5.in_diheral_angles = [-driving_angles]

    lines.append(Line(p1,p2,driving_angles))
    lines.append(Line(p1,p3,-driving_angles))
    lines.append(Line(p1,p4,-driving_angles))
    lines.append(Line(p1,p5,-driving_angles))

    x,y = position_revert(p1.position)
    board[y][x] = 1
    x,y = position_revert(p2.position)
    board[y][x] = 1
    x,y = position_revert(p3.position)
    board[y][x] = 1
    x,y = position_revert(p4.position)
    board[y][x] = 1
    x,y = position_revert(p5.position)
    board[y][x] = 1
    
    boundary_points.append(p2)
    boundary_points.append(p3)
    boundary_points.append(p4)
    boundary_points.append(p5)

    return points, boundary_points, lines


def position_revert(position: np.ndarray) -> np.ndarray:
    print("position",position)
    return np.array([position[0]+BOARD_SIZE//2,BOARD_SIZE//2-position[1]])
    
def point_is_exist(points: list[Point], p: Point,board: np.ndarray) -> bool:
    x,y = position_revert(p.position)
    print("x,y",x,y)
    return board[y][x] != 0

def expand_point(points: list[Point], lines: list[Line], boundary_points: list[Point], p1: Point, root_point_sym: Point, diheral_angles_1: float, board: np.ndarray):
    # global board
    ACTIONS.append(
    {
        "action": Action.EXPAND_POINT,
        "action_data": ActionData((p1.position[0],p1.position[1]))
    })
    global SYMMETRIC_DIR
    if SYMMETRIC_DIR == SymmetricStrategy.No:
        return p1
    elif SYMMETRIC_DIR == SymmetricStrategy.X:
        new_ext_point =  Point(p1.position[0],-p1.position[1],p1.position[2],root_point_sym)
    else:# SYMMETRIC_DIR == SymmetricStrategy.Y:
        new_ext_point =  Point(-p1.position[0],p1.position[1],p1.position[2],root_point_sym)
    # elif SYMMETRIC_DIR == SymmetricStrategy.X_AND_Y:
    lines.append(Line(root_point_sym,new_ext_point,diheral_angles_1))
    new_ext_point.in_diheral_angles = [diheral_angles_1]
    print("new_ext_point",new_ext_point)
    print("expand point",p1)
    if point_is_exist(points,new_ext_point,board):
        print("expand point",p1)
        is_valid_ = is_valid_to_merge(points,boundary_points, p1, new_ext_point, lines)
        exist_point = get_point_with_position(points,new_ext_point.position)
        if not is_valid_:
            print("Error: ext_point is not valid")
            raise ValueError("Error: ext_point is not valid")
        merge_points(points, lines, boundary_points, exist_point, [new_ext_point])
        new_ext_point = exist_point
    else:
        points.append(new_ext_point)
        boundary_points.append(new_ext_point)
        x,y = position_revert(new_ext_point.position)
        board[y][x] = 1
    is_intersec =  check_intersection(points, [(x.p1, x.p2) for x in lines],
                               [new_ext_point], root_point_sym)
    if is_intersec:
        raise ValueError("Error: intersec in expand point")
    return new_ext_point

def expand_symmetric_X(points: list[Point], lines: list[Line], boundary_points: list[Point], pick_point: Point, p1: Point, p2: Point, p3: Point, list_diheral_angles_1: list[float],board: np.ndarray):
    global SYMMETRIC_DIR
    
    if SYMMETRIC_DIR == SymmetricStrategy.No:
        return pick_point, p1, p2, p3, list_diheral_angles_1
    if is_on_symmetric(pick_point):
        return pick_point, p1, p2, p3, list_diheral_angles_1
    ACTIONS.append(
        {
            "action": Action.EXPAND_SYMMETRIC_X
        }
    )    
    pick_point_symmetric = [x for x in points if x.position[0] == pick_point.position[0] and x.position[1] == -pick_point.position[1]][0]
    if pick_point_symmetric is None:
        raise ValueError("Error: pick_point_symmetric is None")
    if pick_point_symmetric in boundary_points:
        boundary_points.remove(pick_point_symmetric)
    p1_symmetric = expand_point(points,lines,boundary_points,p1,pick_point_symmetric,list_diheral_angles_1[0],board)
    p2_symmetric = expand_point(points,lines,boundary_points,p2,pick_point_symmetric,list_diheral_angles_1[1],board)
    p3_symmetric = expand_point(points,lines,boundary_points,p3,pick_point_symmetric,list_diheral_angles_1[2],board)
    # visualize(points,lines,True)
    return pick_point_symmetric, p1_symmetric, p2_symmetric, p3_symmetric, list_diheral_angles_1
        
def expand_symmetric_Y(points: list[Point], lines: list[Line], boundary_points: list[Point], pick_point: Point, p1: Point, p2: Point, p3: Point, list_diheral_angles_1: list[float],board: np.ndarray):
    global SYMMETRIC_DIR
    print("expand_Y")
    if SYMMETRIC_DIR == SymmetricStrategy.No:
        return pick_point, p1, p2, p3, list_diheral_angles_1
    if is_on_symmetric(pick_point):
        return pick_point, p1, p2, p3, list_diheral_angles_1
    ACTIONS.append(
        {
            "action": Action.EXPAND_SYMMETRIC_Y
        }
    )
    pick_point_symmetric = [x for x in points if x.position[1] == pick_point.position[1] and x.position[0] == -pick_point.position[0]][0]
    if pick_point_symmetric is None:
        raise ValueError("Error: pick_point_symmetric is None")
    if pick_point_symmetric in boundary_points:
        boundary_points.remove(pick_point_symmetric)
    p1_symmetric = p1
    p2_symmetric = p2
    p3_symmetric = p3

    p1_symmetric = expand_point(points,lines,boundary_points,p1,pick_point_symmetric,list_diheral_angles_1[0],board)
    p2_symmetric = expand_point(points,lines,boundary_points,p2,pick_point_symmetric,list_diheral_angles_1[1],board)
    p3_symmetric = expand_point(points,lines,boundary_points,p3,pick_point_symmetric,list_diheral_angles_1[2],board)
    # visualize(points,lines,True)
    return pick_point_symmetric, p1_symmetric, p2_symmetric, p3_symmetric, list_diheral_angles_1

def expand_symmetric(points: list[Point], lines: list[Line], boundary_points: list[Point], pick_point: Point, p1: Point, p2: Point, p3: Point, list_diheral_angles_1: list[float],board: np.ndarray):
    global SYMMETRIC_DIR
    if SYMMETRIC_DIR == SymmetricStrategy.No:
        return
    pick_point_symmetric = None
    if SYMMETRIC_DIR == SymmetricStrategy.X:
        pick_point_symmetric, p1_symmetric, p2_symmetric, p3_symmetric, list_diheral_angles_1 = expand_symmetric_X(points,lines,boundary_points,pick_point,p1,p2,p3,list_diheral_angles_1,board)
    elif SYMMETRIC_DIR == SymmetricStrategy.Y:
        pick_point_symmetric, p1_symmetric, p2_symmetric, p3_symmetric, list_diheral_angles_1 = expand_symmetric_Y(points,lines,boundary_points,pick_point,p1,p2,p3,list_diheral_angles_1,board)
    elif SYMMETRIC_DIR == SymmetricStrategy.X_AND_Y:
        try:
            SYMMETRIC_DIR = SymmetricStrategy.X
            pick_point_symmetric_X, p1_symmetric_X, p2_symmetric_X, p3_symmetric_X, list_diheral_angles_1_X = expand_symmetric_X(points,lines,boundary_points,pick_point,p1,p2,p3,list_diheral_angles_1,board)
            SYMMETRIC_DIR = SymmetricStrategy.Y

            expand_symmetric_Y(points,lines,boundary_points,pick_point,p1,p2,p3,list_diheral_angles_1,board)
            expand_symmetric_Y(points,lines,boundary_points,pick_point_symmetric_X,p1_symmetric_X,p2_symmetric_X,p3_symmetric_X,list_diheral_angles_1_X,board)
            SYMMETRIC_DIR = SymmetricStrategy.X_AND_Y
        except Exception as e:
            SYMMETRIC_DIR = SymmetricStrategy.X_AND_Y
            raise e
    
    return pick_point, p1, p2, p3, list_diheral_angles_1

def store_output(data, file_name, folder_path = ""):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if folder_path == "":
        folder_path = os.path.join(BASE_DIR, "output")

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file_name)
    file_str = json.dumps(data, indent=2, cls=AutoEncoder)

    with open(file_path, "w") as f:
        f.write(file_str)

def gen_ptu_board(driving_angles:float,num_in: int, merge_radius: float,id: str, is_visualize = False) -> tuple[list[Point],list[Line]]:
    points:list[Point] = []
    num_loop = 0
    points = []
    lines = []
    boundary_points = []
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    print("OKOKO")
    points, boundary_points, lines = create_base_X_and_Y(points, lines, boundary_points, driving_angles,board)
    print("points:",len(points))
    print("lines:",len(lines))
    print("boundary_points:",len(boundary_points))
    # try:
    while len(points) < num_in and num_loop < 10000:
        num_loop += 1
        ACTIONS.append(
            {
                "action": Action.START
            }
        )
        pick_point = boundary_points[pick_up_point(boundary_points)] 
        p1,p2,p3 = pick_position(pick_point)
        
        p1 = Point(p1[0],p1[1],0,pick_point)
        p2 = Point(p2[0],p2[1],0,pick_point)
        p3 = Point(p3[0],p3[1],0,pick_point)

        temp_points = points + [p1,p2,p3]
        temp_lines = lines + [Line(pick_point,temp_points[-3]),Line(pick_point,temp_points[-2]),Line(pick_point,temp_points[-1])]
        temp_boundary_points = boundary_points.copy()
        temp_boundary_points.remove(pick_point)
        temp_boundary_points.append(p1)
        temp_boundary_points.append(p2)
        temp_boundary_points.append(p3)
        visualize(temp_points,temp_lines,True)

        temp_lines = temp_lines + [Line(x,y,-999) for x,y in connect_boundary_points(temp_points,temp_boundary_points,temp_lines)]
        is_valid_p = is_valid(pick_point,p1,p2,p3,temp_points, temp_boundary_points, temp_lines,board)
        temp_points = points.copy() + [p1,p2,p3]
        temp_lines = lines.copy() + [Line(pick_point,temp_points[-3]),Line(pick_point,temp_points[-2]),Line(pick_point,temp_points[-1])]
        # visualize(temp_points,temp_lines,True)
        if is_valid_p:
            sector_angles, in_diheral_angles, p1, p2, p3  = calc_angles(pick_point,p1,p2,p3)
            print("sector_angles: ", sector_angles, "\n driving angle: ", in_diheral_angles)
            temp_points = points + [p1,p2,p3]
            temp_lines = lines + [Line(pick_point,temp_points[-3]),Line(pick_point,temp_points[-2]),Line(pick_point,temp_points[-1])]

            visualize(temp_points,temp_lines,is_visualize)

            sector_angles_, list_diheral_angles_1, list_diheral_angles_2 = calc_ptu(sector_angles,in_diheral_angles)
        
            if list_diheral_angles_1 == []: 
                continue
            print("list_diheral_angles_1:",list_diheral_angles_1)
            print("list_diheral_angles_2:",list_diheral_angles_2)
            list_diheral_angles_1 = random.choice([list_diheral_angles_1,list_diheral_angles_2])
            list_diheral_angles_1 = list_diheral_angles_2
            print("selected list_diheral_angles:",list_diheral_angles_1)
            temp_lines = lines.copy()
            temp_points = points.copy()
            temp_boundary_points = boundary_points.copy()
            temp_boundary_points.remove(pick_point)

            p1.in_diheral_angles = [list_diheral_angles_1[0]]
            p2.in_diheral_angles = [list_diheral_angles_1[1]]
            p3.in_diheral_angles = [list_diheral_angles_1[2]]

            temp_lines.append(Line(pick_point,p1,list_diheral_angles_1[0]))
            temp_lines.append(Line(pick_point,p2,list_diheral_angles_1[1]))
            temp_lines.append(Line(pick_point,p3,list_diheral_angles_1[2]))
            temp_board = board.copy()
            print("genboard")
            # print("p1:",p1)
            # print(temp_board)
            if point_is_exist(temp_points,p1,temp_board):
                exist_point = get_point_with_position(temp_points,p1.position)
                if p1 == exist_point:
                    print("p1 == exist_point")
                if not is_valid_to_merge(temp_points,temp_boundary_points,exist_point,p1,temp_lines): continue
                merge_points(temp_points, temp_lines, temp_boundary_points, exist_point, [p1])
            else:
                temp_boundary_points.append(p1)
                temp_points.append(p1)
                x,y = position_revert(p1.position)
                temp_board[y][x] = 1
            # print(temp_board)
            # for i in range(len(temp_points)):
                # print(i,temp_points[i])
            if point_is_exist(temp_points,p2,temp_board):
                exist_point = get_point_with_position(temp_points,p2.position)
                if not is_valid_to_merge(temp_points,temp_boundary_points,exist_point,p2,temp_lines): continue
                merge_points(temp_points, temp_lines, temp_boundary_points, exist_point , [p2])
            else:
                temp_boundary_points.append(p2)
                temp_points.append(p2)
                x,y = position_revert(p2.position)
                temp_board[y][x] = 1
            if point_is_exist(temp_points,p3,temp_board):
                exist_point = get_point_with_position(temp_points,p3.position)
                if is_valid_to_merge(temp_points,temp_boundary_points,exist_point,p3,temp_lines): continue
                merge_points(temp_points, temp_lines, temp_boundary_points, exist_point , [p3])
            else:
                temp_boundary_points.append(p3)
                temp_points.append(p3)
                x,y = position_revert(p3.position)
                temp_board[y][x] = 1
            # print(temp_board)
            # for i in range(len(temp_points)):
                # print(i,temp_points[i])
            # print("temp_board",temp_board)
            # visualize(temp_points,temp_lines,is_visualize)
            try:
                expand_symmetric(temp_points,temp_lines,temp_boundary_points,pick_point,p1,p2,p3,list_diheral_angles_1,temp_board)
                boundary_lines = connect_boundary_points(temp_points,temp_boundary_points,temp_lines)
                if len(boundary_lines) < 3 and len(temp_boundary_points) > 2:
                    ACTIONS.append(
                        {
                            "action": Action.DESTROY
                        }
                    )
                    continue
                lines = temp_lines
                points = temp_points
                boundary_points = temp_boundary_points
                board = temp_board
            except Exception as e:
                print("Error: expand_symmetric:",e)
                ACTIONS.append(
                    {
                        "action": Action.DESTROY
                    }
                )
            print("boundary points:",len(temp_boundary_points))
            # for i in points:
                # print("points",i.position)
                # print("root_points",i.point_root)
                # print("in_diheral_angles",i.in_diheral_angles)
            visualize(points,lines,is_visualize)
            ACTIONS.append(
                {
                    "action": Action.END_STEP
                }
            )
            
            # for i in range(len(lines)):
            #     print(lines[i].p1,lines[i].p2)
            # visualize(points,lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)

    ACTIONS.append(
        {
            "action": Action.END
        }
    )
    store_output(ACTIONS, f"actions.json",f"output/output_{id}")
    return points, lines +[Line(x,y,-999) for x,y in connect_boundary_points(points,boundary_points,lines)]
            
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(points, lines, is_visualize = False):
    if not is_visualize:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot all points
    xs = [p.position[0] for p in points]
    ys = [p.position[1] for p in points]
    zs = [p.position[2] for p in points]
    ax.scatter(xs, ys, zs, color='blue', s=50, label='Points')

    # label each point by index
    for i, p in enumerate(points):
        ax.text(p.position[0], p.position[1], p.position[2], f'{i}', color='red')

    # plot all lines
    for line in lines:
        p1 = line.p1.position
        p2 = line.p2.position
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=1)

    # formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Points and Lines Visualization")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()
