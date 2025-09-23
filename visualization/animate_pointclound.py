import torch
import numpy as np
import open3d as o3d
import random
from object.origami_object import OrigamiObject, Point, LineType, Line, Face
from typing import Callable
import copy

class Plotter:
    def __init__(self, o: OrigamiObject, nextFrame: Callable) -> None:
        self.o = o
        self.geometries = {
            "Point": [],
            "Line": [],
            "Face": [],
        } # list of (geometry, base_vertices)
        for _ in range(len(o.listPointCloud)):
            self.geometries["Point"].append(self.init_point(size=0.01))
        # for line in o.listLines:
        #     print(line.get_line_original_length(o.listPoints))
        #     self.geometries["Line"].append(self.init_line(length=line.get_line_original_length(o.listPoints), size=5))
        self.nextFrame = nextFrame

    def show(self):
        vis = o3d.visualization.Visualizer() # type: ignore
        vis.create_window(window_name="Origami Animation", width=1280, height=720)
        
        for g in self.geometries['Point']:
            vis.add_geometry(g[0])
        # for g in self.geometries['Line']:
        #     vis.add_geometry(g[0])
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

        view_ctl = vis.get_view_control()
        view_ctl.set_lookat(np.array([0.0, 0, 0.0]))
        view_ctl.set_up([0, 1, 0])
        view_ctl.set_front([0, 0, -1])

        while True:
            self.nextFrame(self.o)
            self.update_geometries()
            for g in self.geometries['Point']:
                vis.update_geometry(g[0])
            # for g in self.geometries['Line']:
            #     vis.update_geometry(g[0])
            if not vis.poll_events():
                break
            vis.update_renderer()
        
    def update_geometries(self):
        for i, g in enumerate(self.geometries["Point"]):
            self.set_position_point(g[0], g[1], self.o.listPointCloud[i]["point"].position)
        # for i, g in enumerate(self.geometries["Line"]):
        #     p1 = self.o.listPoints[self.o.listLines[i].p1Index].position
        #     p2 = self.o.listPoints[self.o.listLines[i].p2Index].position
        #     self.set_position_line(g[0], g[1], p1, p2)

    def tensor_to_np(self, t: torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float64)

    def draw_axis(self, size=1.0):
        pass
        #self.geometries.append(["Axis",o3d.geometry.TriangleMesh.create_coordinate_frame(size=size), np.array([])])

    def init_point(self, size=0.1, color: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh.paint_uniform_color(color)
        return [
            mesh,
            np.asarray(mesh.vertices).copy()
        ]
    
    def set_position_point(self, mesh: o3d.geometry.TriangleMesh, base_vertices: np.ndarray, position: torch.Tensor):
        p = self.tensor_to_np(position)
        mesh.vertices = o3d.utility.Vector3dVector(base_vertices.copy() + p)
        mesh.compute_vertex_normals()
        
    def init_line(self, length: float, size: float = 0.05, color: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=size, height=length)
        mesh.translate(np.array([0, 0, length/2]))
        return [
            mesh,
            np.asarray(mesh.vertices).copy()
        ]
    
    def set_position_line(self, mesh: o3d.geometry.TriangleMesh, base_vertices: np.ndarray, point1: torch.Tensor, point2: torch.Tensor):
        p1 = self.tensor_to_np(point1)
        p2 = self.tensor_to_np(point2)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        
        mesh.vertices = o3d.utility.Vector3dVector(base_vertices.copy())
        direction /= length
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        
        if np.linalg.norm(rotation_axis) < 1e-6:
            if rotation_angle > 0:
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
            else:
                rotation_matrix = np.identity(3)
        else:
            rotation_vector = rotation_axis / np.linalg.norm(rotation_axis) * rotation_angle
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
            
        mesh.rotate(rotation_matrix, center=np.array([0, 0, 0]))
        mesh.translate(p1)
        mesh.compute_vertex_normals()

    def init_arrow(self, point1: torch.Tensor, point2: torch.Tensor, size: float = 0.05, color: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        pass
        # p1 = self.tensor_to_np(point1)
        # p2 = self.tensor_to_np(point2)
        # direction = p2 - p1
        # length = np.linalg.norm(direction)
        # if length < 1e-6:
        #     return
            
        # arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=size, cone_radius=size*2, cylinder_height=length*0.9, cone_height=length*0.1)
        
        # direction /= length
        # z_axis = np.array([0, 0, 1])
        # rotation_axis = np.cross(z_axis, direction)
        # rotation_angle = np.arccos(np.dot(z_axis, direction))
        
        # if np.linalg.norm(rotation_axis) < 1e-6:
        #     if rotation_angle > 0:
        #         rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
        #     else:
        #         rotation_matrix = np.identity(3)
        # else:
        #     rotation_vector = rotation_axis / np.linalg.norm(rotation_axis) * rotation_angle
        #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
            
        # arrow.rotate(rotation_matrix, center=np.array([0, 0, 0]))
        # arrow.translate(p1)
        # arrow.paint_uniform_color(color)
        # arrow.compute_vertex_normals()
        # self.geometries.append(arrow)


    def transform_cylinder(self, mesh, p1, p2, p3, p4, eps=1e-9):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        p3 = np.asarray(p3, dtype=float)
        p4 = np.asarray(p4, dtype=float)
        v1 = p2 - p1
        v2 = p4 - p3
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 < eps or l2 < eps:
            raise ValueError("Endpoint pairs must be distinct non-zero length")
        u = v1 / l1
        v = v2 / l2
        dot = np.clip(np.dot(u, v), -1.0, 1.0)
        if dot > 1 - 1e-12:
            R = np.eye(3)
        elif dot < -1 + 1e-12:
            arb = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(arb, u)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            axis = np.cross(u, arb)
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        else:
            axis = np.cross(u, v)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        s = l2 / l1
        c1 = (p1 + p2) / 2.0
        c2 = (p3 + p4) / 2.0
        MSR = np.eye(4)
        MSR[:3, :3] = s * R
        T1 = np.eye(4)
        T1[:3, 3] = -c1
        T2 = np.eye(4)
        T2[:3, 3] = c2
        transform = T2 @ MSR @ T1
        new_mesh = copy.deepcopy(mesh)
        new_mesh.transform(transform)
        return new_mesh
