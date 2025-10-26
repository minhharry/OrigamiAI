import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from object.origami_object import Point, Line, Face
import time
import math

def dd(*x):
    if True:
        print("---- DEBUG: ", *x)

def convert_to_matrix(listPoints: list[Point], listLines: list[Line], listFaces: list[Face]) -> dict:
    pointTensors = []
    for point in listPoints:
        pointTensors.append(point.position)
    resPoints = torch.stack(pointTensors)
    lineIndices = []
    for line in listLines:
        lineIndices.append([line.p1Index, line.p2Index])
    resLines = torch.tensor(lineIndices)
    
    mappingLineToFace = {}
    for face in listFaces:
        face.calculate_and_update_line_index(listLines)
        if mappingLineToFace.get(face.line12Index) is None:
            mappingLineToFace[face.line12Index] = []
        if mappingLineToFace.get(face.line23Index) is None:
            mappingLineToFace[face.line23Index] = []
        if mappingLineToFace.get(face.line13Index) is None:
            mappingLineToFace[face.line13Index] = []
        mappingLineToFace[face.line12Index].append(face)
        mappingLineToFace[face.line23Index].append(face)
        mappingLineToFace[face.line13Index].append(face)
    
    resFaces = []
    resTargetThetas = []
    for lineIndex in range(len(listLines)):
        if mappingLineToFace.get(lineIndex) is None:
            continue
        if len(mappingLineToFace[lineIndex]) != 2:
            continue
        face1: Face = mappingLineToFace[lineIndex][0]
        face2: Face = mappingLineToFace[lineIndex][1]
        idx1, idx2 = listLines[lineIndex].p1Index, listLines[lineIndex].p2Index
        if idx1 == face1.point1Index and idx2 == face1.point2Index \
           or idx1 == face1.point2Index and idx2 == face1.point3Index \
           or idx1 == face1.point3Index and idx2 == face1.point1Index:
            pass
        else:
            idx1, idx2 = idx2, idx1

        tmp = []

        if idx1 != face2.point1Index and idx2 != face2.point1Index:
            tmp.append(face2.point1Index)
        elif idx1 != face2.point2Index and idx2 != face2.point2Index:
            tmp.append(face2.point2Index)
        elif idx1 != face2.point3Index and idx2 != face2.point3Index:
            tmp.append(face2.point3Index)
        
        if idx1 != face1.point1Index and idx2 != face1.point1Index:
            tmp.append(face1.point1Index)
        elif idx1 != face1.point2Index and idx2 != face1.point2Index:
            tmp.append(face1.point2Index)
        elif idx1 != face1.point3Index and idx2 != face1.point3Index:
            tmp.append(face1.point3Index)  
        tmp.append(idx1)
        tmp.append(idx2)
        resFaces.append(tmp)
        resTargetThetas.append([listLines[lineIndex].targetTheta])
    resFaces = torch.tensor(resFaces)
    resTargetThetas = torch.tensor(resTargetThetas)
    return {
        "points": resPoints,
        "lines": resLines,
        "faces": resFaces,
        "target_thetas": resTargetThetas,
    }

class OrigamiObjectMatrix:
    def __init__(self, points: torch.Tensor,
                 lines: torch.Tensor,
                 faces: torch.Tensor,
                 target_theta: torch.Tensor,
                 mass: float = 1.0,
                 ea: float = 20.0,
                 k_crease: float = 0.7,
                 damping: float = 0.45,
                 fold_percent: float = 0.99,
                 dt: float = -1.0, # -1 means auto config 
                 device: str = 'cpu'):
        self.device = device
        self.points = points.to(self.device)
        self.lines = lines.to(self.device)
        self.faces = faces.to(self.device)

        self.num_points = points.shape[0]
        self.num_lines = lines.shape[0]
        self.num_faces = faces.shape[0]
        self.masses = mass
        self.ea = ea
        self.k_crease = k_crease
        self.theta = torch.full((self.num_faces, 1), 0.0, device=self.device)
        self.target_theta = (target_theta*fold_percent).to(self.device)
        self.origin_length = torch.zeros_like(self.target_theta)
        for i in range(self.num_faces):
            self.origin_length[i] += torch.linalg.norm(self.points[self.faces[i, 2]] - self.points[self.faces[i, 3]])

        self.damping = damping
        self.velocities = torch.zeros_like(self.points)
        self.total_forces = torch.zeros_like(self.points) # Added for visualization

        p1 = self.points[self.lines[:, 0]]
        p2 = self.points[self.lines[:, 1]]
        self.rest_lengths = torch.norm(p2 - p1, dim=1, keepdim=True)

        self.dt = dt
        if dt < 0.0:
            self.dt = 1.0/(2.0*math.pi*math.sqrt(self.ea/self.rest_lengths.min().item()))
    
    def step(self):
        point1 = self.points[self.lines[:, 0]]
        point2 = self.points[self.lines[:, 1]]

        spring_vectors = point2 - point1
        current_lengths = torch.linalg.norm(spring_vectors, dim=1, keepdim=True)
        current_lengths[current_lengths <= 1e-6] = 1e-6
        displacements = current_lengths - self.rest_lengths
        force_magnitudes = -self.ea / self.rest_lengths * displacements
        force_vectors = force_magnitudes * (spring_vectors / current_lengths)

        self.total_forces = torch.zeros_like(self.points) # Reset for each step
        self.total_forces.index_add_(0, self.lines[:, 1], force_vectors)  # spring force
        self.total_forces.index_add_(0, self.lines[:, 0], -force_vectors) # spring force

        # fold force
        p1 = self.points[self.faces[:, 0]]
        p2 = self.points[self.faces[:, 1]]
        p3 = self.points[self.faces[:, 2]]
        p4 = self.points[self.faces[:, 3]]

        vec_p3p1 = p1 - p3
        vec_p3p2 = p2 - p3
        vec_p3p4 = p4 - p3
        vec_n1 = torch.cross(vec_p3p1, vec_p3p4, dim=1)
        vec_n2 = torch.cross(vec_p3p4, vec_p3p2, dim=1)
        vec_n1_length = torch.linalg.norm(vec_n1, dim=1, keepdim=True)
        vec_n2_length = torch.linalg.norm(vec_n2, dim=1, keepdim=True)
        vec_n1_length[vec_n1_length <= 1e-6] = 1e-6
        vec_n2_length[vec_n2_length <= 1e-6] = 1e-6

        vec_n1 /= vec_n1_length
        vec_n2 /= vec_n2_length

        h1 = vec_n1_length / torch.linalg.norm(vec_p3p4, dim=1, keepdim=True)
        h2 = vec_n2_length / torch.linalg.norm(vec_p3p4, dim=1, keepdim=True)

        dotNormals = (vec_n1 * vec_n2).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        cross_n1_crease = torch.linalg.cross(vec_n1, vec_n2, dim=1)
        y = (cross_n1_crease * vec_p3p4).sum(dim=1, keepdim=True)
        unsignedTheta = torch.acos(dotNormals)  
        signTheta = torch.sign(y)
        theta = unsignedTheta * signTheta
        diff = theta - self.theta
        add_tensor = diff + 2 * torch.pi
        sub_tensor = diff - 2 * torch.pi
        diff = torch.where(diff < -5.0, add_tensor, diff)
        diff = torch.where(diff > 5.0, sub_tensor, diff)
        theta = self.theta + diff
        self.theta = theta
        force_magnitudes_2 = -self.k_crease * self.origin_length * (theta - (self.target_theta))

        p1_forces_vectors = force_magnitudes_2 * (vec_n1 / h1)
        p2_forces_vectors = force_magnitudes_2 * (vec_n2 / h2)        
        p3_forces_vectors = p1_forces_vectors + p2_forces_vectors
        p3_forces_vectors = -p3_forces_vectors/2

        self.total_forces.index_add_(0, self.faces[:, 0], p1_forces_vectors)
        self.total_forces.index_add_(0, self.faces[:, 1], p2_forces_vectors)
        self.total_forces.index_add_(0, self.faces[:, 2], p3_forces_vectors)
        self.total_forces.index_add_(0, self.faces[:, 3], p3_forces_vectors)
        
        # damping force
        point1velocity = self.velocities[self.lines[:, 0]]
        point2velocity = self.velocities[self.lines[:, 1]]
        damping_force =  2.0*self.damping*torch.sqrt(self.ea/self.rest_lengths) * (point2velocity - point1velocity)
        self.total_forces.index_add_(0, self.lines[:, 0], damping_force)
        self.total_forces.index_add_(0, self.lines[:, 1], -damping_force)

        accelerations = self.total_forces / self.masses
        self.velocities += accelerations * self.dt
        self.points += self.velocities * self.dt



if __name__ == "__main__":
    points = torch.tensor([
        [-1., 0., 0.], [0.3, -2., 0.], [1., 0., 0.], [-0.5, 1.5, 0.],
    ])

    lines = torch.tensor([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [2, 3],
    ])

    faces_indices = torch.tensor([
        [3, 1, 2, 0]
    ])
    target_theta = torch.tensor([
        [3.10],
    ])

    ori = OrigamiObjectMatrix(points, lines, faces_indices, target_theta)

    VISUALIZE = True
    if VISUALIZE:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        fig.show()

        keep_running = True
        def on_close(event):
            global keep_running
            keep_running = False
        fig.canvas.mpl_connect('close_event', on_close)

    start_time = time.time()

    for i in range(10000):
        ori.step()
        if VISUALIZE and i % 100 == 0:
            points = ori.points.cpu().numpy()
            lines = ori.lines.cpu().numpy()
            ax.clear()
            for line in lines:
                p1 = points[line[0]]
                p2 = points[line[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-')

            # Draw points (vertices)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=20) # type: ignore

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(f'(Frame {i})')
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            # time.sleep(0.1)
            if not keep_running:
                break


    dd('Run time: ', time.time() - start_time)

