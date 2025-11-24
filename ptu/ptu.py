from __future__ import annotations
import numpy as np
import random
from scipy.spatial import ConvexHull
import math
from enum import Enum
import alphashape
from shapely.geometry import Polygon, LineString, MultiLineString, Point as ShapelyPoint


NUM_IN = [1,0,0]
DEGREE = NUM_IN[0] + NUM_IN[1] + NUM_IN[2] + 3
EPS = 1e-5
GRID_SIZE = 10.0

class Strategy(Enum):
    RANDOM = 1
    BFS = 2
    DFS = 3
    EVO = 4
    PPO = 5

ptu_strategy = Strategy.RANDOM

class Point:
    def __init__(self, x: float, y: float, z: float, point_root: Point):
        self.position = np.array([x,y,z])
        self.in_diheral_angles = []
        self.out_diheral_angles = []
        self.point_root = [point_root]
    
    def __str__(self):
        return f"Point({self.position[0]},{self.position[1]},{self.position[2]})"

def gen_sector_angles(point: Point)->tuple[list[list[float]],list[list[float]]]:
    sector_angles = [[np.pi/2+EPS,np.pi/2-EPS],[np.pi/2+EPS],[np.pi/2-EPS]] 
    angles1 = random.random()*np.pi*2
    angles2 = random.random()*np.pi*2
    angles3 = random.random()*np.pi*2
    root = point.position+np.array([0,0,1])

    return sector_angles,  [[point.in_diheral_angles[0]],[],[]]

class Line:
    def __init__(self, p1: Point, p2: Point, targetTheta: float = 0):
        self.p1 = p1
        self.p2 = p2
        self.targetTheta = targetTheta

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

            m_revert = np.identity(3)    
            m_revert = np.matmul(Rz(-sector_angles[index][-1]),m_revert)
            alpha_vec_revert = sector_angles[index].copy()
            alpha_vec_revert = alpha_vec_revert[:-1]
            list_in_diheral_angles_revert = list_in_diheral_angles[index].copy()
            list_in_diheral_angles_revert.reverse()
            for i in range(m-1):
                m_revert = np.matmul(m_revert,transform_fold_rev(-list_in_diheral_angles_revert[i], -alpha_vec_revert[i]))

            p_j_0_revert = np.matmul(m_revert,p0)
            p_jm1_revert = np.matmul(Rz(-sector_angles[index][-1]),p0)
            delta = beta_delta(p_jm1_revert, p_j_0_revert,u,sector_angles[index][0])
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
    if (arr_u[0]+arr_u[1]>=arr_u[2]):
        pass
    else:
        print("KHONG GAP DUOC DAU HEHE ====================")
        return [],[]
    beta1,delta1 = calc_beta_delta(0)
    beta2,delta2 = calc_beta_delta(1)
    beta3,delta3 = calc_beta_delta(2)

    theta1, theta2, theta3 = calculate_theta(u1,u2,u3)

    phi1 = beta3 + np.pi - theta1 + delta2
    phi2 = beta1 + np.pi - theta2 + delta3
    phi3 = beta2 + np.pi - theta3 + delta1

    M1:list[float] = [phi1,phi2,phi3]

    phi1 = beta3 + theta1 - np.pi + delta2
    phi2 = beta1 + theta2 - np.pi + delta3
    phi3 = beta2 + theta3 - np.pi + delta1

    M2:list[float] = [phi1,phi2,phi3]
    return M1,M2

def calc_ptu(sector_angles:list[list[float]],list_in_diheral_angles:list[list[float]]) -> list[list[float]]:
   if (not sector_angles or not list_in_diheral_angles):
       print(len(sector_angles),len(list_in_diheral_angles))
       print("Error: sector_angles or list_in_diheral_angles is empty")
       return [[],[],[]]

   calc(sector_angles,list_in_diheral_angles)
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
    if ptu_strategy == Strategy.BFS:
        return 0
    elif ptu_strategy == Strategy.DFS:
        return len(boundary_points)-1
    return random.randrange(0,len(boundary_points))

def connect_boundary_points(boundary_points: list[Point], list_lines: list[Line]) -> list[tuple[Point, Point]]:
    if len(boundary_points) < 2:
        return []
    
    pts_2d = np.array([[p.position[0], p.position[1]] for p in boundary_points])
    centroid = np.mean(pts_2d, axis=0)

    angles = np.arctan2(pts_2d[:, 1] - centroid[1], pts_2d[:, 0] - centroid[0])

    order = np.argsort(angles)

    boundary_lines = []
    for i in range(len(order)):
        a = boundary_points[order[i]]
        b = boundary_points[order[(i + 1) % len(order)]]
        boundary_lines.append((a, b))

    return boundary_lines

# def connect_boundary_points(boundary_points: list[Point], list_current_lines: list[Line]):
#     if len(boundary_points) < 3:
#         return []
#     for i in range(len(boundary_points)):
#         print(boundary_points[i].position)
#     for i in range(len(list_current_lines)):
#         print(list_current_lines[i].p1.position,list_current_lines[i].p2.position)
#     # Lấy tất cả các điểm có trong system (cả line và boundary)
#     all_points = set(boundary_points)
#     for line in list_current_lines:
#         a, b = line.p1, line.p2
#         all_points.add(a)
#         all_points.add(b)
#     pts_2d = np.array([[p.position[0], p.position[1]] for p in all_points])
#     print("All points (for alpha shape):")
#     for p in pts_2d:
#         print(p)
#     print("Number of points:", len(pts_2d))

#     # Tạo alpha shape bao trùm toàn bộ điểm
#     alpha = 2.0   # điều chỉnh: nhỏ quá thì cắt, lớn quá thì thành convex
#     shape = alphashape.alphashape(pts_2d, alpha)
#     print("shape ",shape)
#     if not isinstance(shape, Polygon):
#         return []

#     coords = np.array(shape.exterior.coords[:-1])  # bỏ điểm lặp cuối

#     # Lọc ra những Point nào nằm trên boundary thật
#     boundary_lines = []
#     ordered_points = []
#     for coord in coords:
#         p = min(boundary_points, key=lambda q: np.linalg.norm(np.array(q.position[:2]) - coord))
#         ordered_points.append(p)

#     # Tạo các cặp cạnh
#     for i in range(len(ordered_points)):
#         a = ordered_points[i]
#         b = ordered_points[(i + 1) % len(ordered_points)]
#         boundary_lines.append((a, b))

#     return boundary_lines


def check_intersection(list_points: list[Point], list_lines: list[tuple[Point,Point]], new_points: list[Point], v_i: Point ) -> bool:

# vi - vp = -s(vN+j − vi) + t(vq − vp)
    if len(list_points) <= 3:
        return False
    list_lines_ :list[tuple[Point,Point]] = list_lines.copy()
    for i in range(len(new_points)):
        v_i_2 = v_i.position[:2]
        v_N_j = new_points[i].position[:2]
        print("================")
        s_vec = v_N_j - v_i_2 # vNj - vi            
        for j in range(len(list_lines_)):
            if list_lines_[j][0] == new_points[i] or list_lines_[j][1] == new_points[i]:
                continue
            v_p = list_lines_[j][0].position[:2] 
            v_q = list_lines_[j][1].position[:2]
            print("v_i ",v_i_2," v_N_j: ",v_N_j," v_p: ",v_p," v_q: ",v_q)
            t_vec = v_q - v_p
            A = np.column_stack((t_vec, -s_vec))
            b = v_i_2 - v_p
            det = np.linalg.det(A)
            if abs(det) < EPS:
                continue 

            s, t = np.linalg.solve(A, b)
            print("s: ",s," t ",t)
            if 0-EPS <= s <= 1+EPS and 0-EPS <= t <= 1+EPS:
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


def is_in_polygon(merge_point: Point, consider_point: Point,  points: list[Point], lines: list[tuple[Point,Point]]) -> bool:
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

def check_in_r(points: list[Point], boundary_points: list[Point], v:Point ,merge_radius: float) -> tuple[bool, list[Point]]:
    check_points = v
    list_merge_points = []
    for i in range(len(boundary_points)):
        if boundary_points[i] == check_points: continue
        if  distance(check_points.position, boundary_points[i].position) < merge_radius and \
            check_points.point_root[0] != boundary_points[i].point_root[0] and \
            True:
            # not is_in_polygon(check_points,boundary_points[i].point_root[0].position):
            list_merge_points.append(boundary_points[i])
            break
    return len(list_merge_points) > 0, list_merge_points
#chua xet lai base point sau khi merge

def merge_points(points: list[Point], lines: list[tuple[Point,Point]], boundary_points: list[Point], v:Point, list_merge_points: list[Point]):
    # list point 
    # list line
    # 
    merge_point = v
    for i in list_merge_points:
        boundary_points.remove(i)
    
    for i in list_merge_points[::-1]:
        points.remove(i)
    
    for i in range(len(lines)):
        if lines[i][0] in list_merge_points:
            lines[i] = (merge_point,lines[i][1])
        elif lines[i][1] in list_merge_points:
            lines[i] = (lines[i][0],merge_point)

#chua kiem tra
    merge_root_point = merge_point.point_root
    list_merge_root_point = list_merge_points[0].point_root
    list_driving_angles = [merge_point.in_diheral_angles+list_merge_points[0].in_diheral_angles]
    sorted_point, sorted_driving_angle = sort_points_ccw(merge_root_point + list_merge_root_point,list_driving_angles, merge_point)

    merge_point.point_root = sorted_point
    merge_point.in_diheral_angles =  sorted_driving_angle


def create_points(root_point: Point, points: list[Point], sector_angles: list[list[float]], list_diheral_angles: list[float]) -> list[Point]:

    root_point = root_point
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

# tam thoi chi xet 1 nghiem
# chua xu ly duyet xong con boundary thi lam gi, hinh vuong
def gen_ptu(driving_angles:float,num_in: int, merge_radius: float, is_visualize = False) -> tuple[list[Point],list[Line]]:
    points:list[Point] = []
    # [Point(-GRID_SIZE/2.0,0.0,GRID_SIZE/2.0),
    #           Point(GRID_SIZE/2.0,0.0,GRID_SIZE/2.0),
    #           Point(GRID_SIZE/2.0,0.0,-GRID_SIZE/2.0),
    #           Point(-GRID_SIZE/2.0,0.0,-GRID_SIZE/2.0)]
    root_point = Point(-GRID_SIZE/2.0,0,0,None)
    points.append(root_point)
    lines:list[Line] = []
    boundary_points:list[Point] = []
    # sinh ra 1 dinh tu root
    root_edge_len = 1
    init_point_position = np.dot(Rz(driving_angles),root_point.position)*root_edge_len
    init_point = Point(init_point_position[0],init_point_position[1],init_point_position[2],root_point)
    points.append(init_point)
    lines.append(Line(points[-1],points[-2],driving_angles))
    boundary_points.append(points[-1])
    init_point.point_root = [points[- 2]]
    init_point.in_diheral_angles = [driving_angles]
    num_loop = 0
    while len(points) < num_in and num_loop < 100:
        pickup_point = boundary_points[pick_up_point(boundary_points)]
        can_merge , list_merge_points = check_in_r(points, boundary_points, pickup_point, merge_radius)
        num_loop += 1
        boundary_lines = connect_boundary_points(boundary_points+[root_point],lines)
        if can_merge:
            visualize(points,lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)
            merge_points(points, [(x.p1,x.p2) for x in lines], boundary_points, pickup_point, list_merge_points)
            visualize(points,lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)
            continue
        else:
            sector_angles, in_diheral_angles  = gen_sector_angles(pickup_point)
            sector_angles_, list_diheral_angles_1, list_diheral_angles_2 = calc_ptu(sector_angles,in_diheral_angles)
            
            list_diheral_angles_1 = random.choice([list_diheral_angles_1,list_diheral_angles_2])
            if list_diheral_angles_1 != []:                
                list_diheral_angles = in_diheral_angles[0] + [list_diheral_angles_1[0]] +\
                                        in_diheral_angles[1] + [list_diheral_angles_1[1]] +\
                                        in_diheral_angles[2] + [list_diheral_angles_1[2]]
                p1,p2,p3 = create_points(pickup_point,points,sector_angles,list_diheral_angles)
                
                new_points = [p1,p2,p3]
                temp_points = points + new_points
                temp_lines = lines + [Line(pickup_point,temp_points[-3]),Line(pickup_point,temp_points[-2]),Line(pickup_point,temp_points[-1])]
                temp_boundary_points = boundary_points.copy()
                temp_boundary_points.remove(pickup_point)
                boundary_lines = connect_boundary_points(temp_boundary_points+new_points+[root_point],lines)
                is_intersection = check_intersection(points,[(x.p1,x.p2) for x in lines]+boundary_lines,new_points,pickup_point)
                visualize(temp_points,temp_lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)
                
                if is_intersection:
                    continue
                else:
                    is_1_in_r,_ = check_in_r(points, points, p1, merge_radius)
                    is_2_in_r,_ = check_in_r(points, points, p2, merge_radius)
                    is_3_in_r,_ = check_in_r(points, points, p3, merge_radius)
                    if is_1_in_r or is_2_in_r or is_3_in_r:
                        continue
                    points.append(p1)
                    points.append(p2)
                    points.append(p3)
               
                    lines.append(Line(pickup_point,points[-3],list_diheral_angles_1[0]))
                    lines.append(Line(pickup_point,points[-2],list_diheral_angles_1[1]))
                    lines.append(Line(pickup_point,points[-1],list_diheral_angles_1[2]))
                    boundary_points.remove(pickup_point)
                    boundary_points.append(points[-3])
                    boundary_points.append(points[-2])
                    boundary_points.append(points[-1])
            else:
                raise ValueError("Error: list_diheral_angles_1 is empty")
            # if list_diheral_angles_2 != []:
    return points, lines +[Line(x,y,-999) for x,y in connect_boundary_points(boundary_points+[root_point],lines)]
            
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



def gen_ptu_square(driving_angles:float, merge_radius: float, is_visualize = False) -> tuple[list[Point],list[Line]]:
    points:list[Point] = []
    # [Point(-GRID_SIZE/2.0,0.0,GRID_SIZE/2.0),
    #           Point(GRID_SIZE/2.0,0.0,GRID_SIZE/2.0),
    #           Point(GRID_SIZE/2.0,0.0,-GRID_SIZE/2.0),
    #           Point(-GRID_SIZE/2.0,0.0,-GRID_SIZE/2.0)]
    root_point = Point(-GRID_SIZE/2.0,0,0,None)
    points.append(root_point)
    lines:list[Line] = []
    boundary_points:list[Point] = []
    # sinh ra 1 dinh tu root
    root_edge_len = 1
    init_point_position = np.dot(Rz(driving_angles),root_point.position)*root_edge_len
    init_point = Point(init_point_position[0],init_point_position[1],init_point_position[2],root_point)
    points.append(init_point)
    lines.append(Line(points[-1],points[-2],driving_angles))
    boundary_points.append(points[-1])
    init_point.point_root = [points[- 2]]
    init_point.in_diheral_angles = [driving_angles]
    num_loop = 0
    while len(boundary_points) != 0:
        pickup_point = boundary_points[pick_up_point(boundary_points)]
        can_merge , list_merge_points = check_in_r(points, boundary_points, pickup_point, merge_radius)
        num_loop += 1
        boundary_lines = connect_boundary_points(boundary_points+[root_point],lines)
        if can_merge:
            visualize(points,lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)
            merge_points(points, [(x.p1,x.p2) for x in lines], boundary_points, pickup_point, list_merge_points)
            visualize(points,lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)
            continue
        else:
            sector_angles, in_diheral_angles  = gen_sector_angles(pickup_point)
            sector_angles_, list_diheral_angles_1, list_diheral_angles_2 = calc_ptu(sector_angles,in_diheral_angles)
            
            list_diheral_angles_1 = random.choice([list_diheral_angles_1,list_diheral_angles_2])
            if list_diheral_angles_1 != []:                
                list_diheral_angles = in_diheral_angles[0] + [list_diheral_angles_1[0]] +\
                                        in_diheral_angles[1] + [list_diheral_angles_1[1]] +\
                                        in_diheral_angles[2] + [list_diheral_angles_1[2]]
                p1,p2,p3 = create_points(pickup_point,points,sector_angles,list_diheral_angles)
                
                new_points = [p1,p2,p3]
                temp_points = points + new_points
                temp_lines = lines + [Line(pickup_point,temp_points[-3]),Line(pickup_point,temp_points[-2]),Line(pickup_point,temp_points[-1])]
                temp_boundary_points = boundary_points.copy()
                temp_boundary_points.remove(pickup_point)
                boundary_lines = connect_boundary_points(temp_boundary_points+new_points+[root_point],lines)
                is_intersection = check_intersection(points,[(x.p1,x.p2) for x in lines]+boundary_lines,new_points,pickup_point)
                visualize(temp_points,temp_lines+[Line(x,y,-999) for x,y in boundary_lines],is_visualize)
                
                if is_intersection:
                    continue
                else:
                    is_1_in_r,_ = check_in_r(points, points, p1, merge_radius)
                    is_2_in_r,_ = check_in_r(points, points, p2, merge_radius)
                    is_3_in_r,_ = check_in_r(points, points, p3, merge_radius)
                    if is_1_in_r or is_2_in_r or is_3_in_r:
                        continue
                    points.append(p1)
                    points.append(p2)
                    points.append(p3)
               
                    lines.append(Line(pickup_point,points[-3],list_diheral_angles_1[0]))
                    lines.append(Line(pickup_point,points[-2],list_diheral_angles_1[1]))
                    lines.append(Line(pickup_point,points[-1],list_diheral_angles_1[2]))
                    boundary_points.remove(pickup_point)
                    boundary_points.append(points[-3])
                    boundary_points.append(points[-2])
                    boundary_points.append(points[-1])
            else:
                raise ValueError("Error: list_diheral_angles_1 is empty")
            # if list_diheral_angles_2 != []:
    return points, lines +[Line(x,y,-999) for x,y in connect_boundary_points(boundary_points+[root_point],lines)]


# points, lines = gen_ptu(np.pi-0.01,9,5)
# visualize(points, lines)