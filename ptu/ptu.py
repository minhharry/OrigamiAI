import numpy as np
import random

NUM_IN = [1,0,0]
DEGREE = NUM_IN[0] + NUM_IN[1] + NUM_IN[2] + 3
EPS = 1e-5
GRID_SIZE = 10.0

class Point:
    def __init__(self, x: float, y: float, z: float, point_root: int = -1):
        self.position = np.array([x,y,z])
        self.in_diheral_angles = []
        self.out_diheral_angles = []
        self.point_root = [point_root]

def gen_sector_angles()->list[list[float]]: 
    sector_angles = [[np.pi/2,np.pi/4],[np.pi/2],[np.pi*3/4]] 
    return sector_angles 

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

def calc(sector_angles,list_in_diheral_angles) -> tuple[list[float],list[float]]:
    
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
   t = [x for y in sector_angles for x in y]
   return [t, M1,M2]
   diheral_angles1 = list_in_diheral_angles[0] + [float(M1[2])] + \
                    list_in_diheral_angles[1] + [float(M1[0])] + \
                    list_in_diheral_angles[2] + [float(M1[1])] 

   diheral_angles2 = list_in_diheral_angles[0] + [float(M2[2])] + \
                    list_in_diheral_angles[1] + [float(M2[0])] + \
                    list_in_diheral_angles[2] + [float(M2[1])] 

   M1 =  diheral_angles1
   M2 = [x for y in sector_angles for x in y], diheral_angles2
   return M1, M2

def pick_up_point(boundary_points: list[int]):
    return random.randrange(0,len(boundary_points))

def check_intersection(list_points: list[Point]) -> bool:
    return False

def check_merge_condition(points: list[Point], lines: list[tuple[int,int]], v_index:int ,merge_radius: float) -> bool:
    return False

def merge_points(points: list[np.ndarray], lines: list[tuple[int,int]], boundary_points_index: list[int], v_index:int, merge_radius: float):
    pass    

def create_points(root_point_index: int, points: list[Point], sector_angles: list[list[float]], list_diheral_angles: list[float]) -> list[Point]:

    root_point = points[root_point_index]
    base_point = points[root_point.point_root[0]]
    p_tuong_doi = base_point.position - root_point.position

    p1 = calc_p_j_m(p_tuong_doi,sector_angles[0][1:],[0^(len(sector_angles[0][1:]))])
    p2 = calc_p_j_m(p1,sector_angles[1],[0^(len(sector_angles[1]))])
    p3 = calc_p_j_m(p2,sector_angles[2],[0^(len(sector_angles[2]))])

    p1 = p1 * random.randrange(1,3)/2.0
    p2 = p2 * random.randrange(1,3)/2.0
    p3 = p3 * random.randrange(1,3)/2.0

    p1 = p1 + root_point.position
    p2 = p2 + root_point.position   
    p3 = p3 + root_point.position

    point1 = Point(p1[0],p1[1],p1[2])
    point2 = Point(p2[0],p2[1],p2[2])
    point3 = Point(p3[0],p3[1],p3[2])

    point1.in_diheral_angles = [[list_diheral_angles[0]],[],[]]
    point2.in_diheral_angles = [[list_diheral_angles[1]],[],[]]
    point3.in_diheral_angles = [[list_diheral_angles[2]],[],[]]

    point1.point_root = [root_point_index]
    point2.point_root = [root_point_index]
    point3.point_root = [root_point_index]

    return [point1,point2,point3]

# tam thoi chi xet 1 nghiem
# chua xu ly duyet xong con boundary thi lam gi, hinh vuong
def gen_ptu(driving_angles:float,num_in: int, merge_radius: float):
    points = []
    # [Point(-GRID_SIZE/2.0,0.0,GRID_SIZE/2.0),
    #           Point(GRID_SIZE/2.0,0.0,GRID_SIZE/2.0),
    #           Point(GRID_SIZE/2.0,0.0,-GRID_SIZE/2.0),
    #           Point(-GRID_SIZE/2.0,0.0,-GRID_SIZE/2.0)]
    root_point = Point(-GRID_SIZE/2.0,0,0)
    points.append(root_point)
    lines = [] #(0,1),(1,2),(2,3),(3,4),(4,0)
    boundary_points_index = [] 
    # sinh ra 1 dinh tu root
    root_edge_len = 1
    init_point_position = np.dot(Rz(driving_angles),root_point.position)*root_edge_len
    init_point = Point(init_point_position[0],init_point_position[1],init_point_position[2])
    print(init_point)
    points.append(init_point)
    lines.append((len(points)-1,len(points)-2))
    boundary_points_index.append(len(points)-1)
    init_point.point_root = [len(points) - 2]
    init_point.in_diheral_angles = [[driving_angles],[],[]]

    while len(points) < num_in:
        pickup_point_index = boundary_points_index[pick_up_point(boundary_points_index)]
        can_merge = check_merge_condition(points, lines, pickup_point_index, merge_radius)
        if can_merge:
            continue
        else:
            pickup_point = points[pickup_point_index]
            sector_angles = gen_sector_angles()
            sector_angles_, list_diheral_angles_1, list_diheral_angles_2 = calc_ptu(sector_angles,pickup_point.in_diheral_angles)
            if list_diheral_angles_1 != []:
                p1,p2,p3 = create_points(pickup_point_index,points,sector_angles,list_diheral_angles_1)
                temp_points = points + [p1,p2,p3]
                is_intersection = check_intersection(temp_points)
                if is_intersection:
                    continue
                else:
                    points.append(p1)
                    points.append(p2)
                    points.append(p3)
               
                    lines.append((pickup_point_index,len(points)-3))
                    lines.append((pickup_point_index,len(points)-2))
                    lines.append((pickup_point_index,len(points)-1))
                    boundary_points_index.remove(pickup_point_index)
                    boundary_points_index.append(len(points)-3)
                    boundary_points_index.append(len(points)-2)
                    boundary_points_index.append(len(points)-1)
                   
            # if list_diheral_angles_2 != []:
    return points, lines
            
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(points, lines):
    """
    Visualize all 3D points and lines using matplotlib.
    - `points`: list of Point objects (each has .position)
    - `lines`: list of (i, j) index pairs connecting points
    """
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
    for (i, j) in lines:
        p1 = points[i].position
        p2 = points[j].position
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


points, lines = gen_ptu(np.pi/2,9,0.1)
visualize(points, lines)