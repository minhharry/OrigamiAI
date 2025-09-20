import torch
import numpy as np
import open3d as o3d
import random

class Plotter:
    def __init__(self) -> None:
        self.geometries = []

    def show(self):
        o3d.visualization.draw_geometries(self.geometries) # type: ignore

    def tensor_to_np(self, t: torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float64)

    def draw_axis(self, size=1.0):
        self.geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=size))

    def draw_point(self, p: torch.Tensor, size=0.1, color: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        np_p = self.tensor_to_np(p)
        self.geometries.append(o3d.geometry.TriangleMesh.create_sphere(radius=size).translate(np_p).paint_uniform_color(color))
        
    def draw_line(self, point1: torch.Tensor, point2: torch.Tensor, size: float = 0.05, color: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        p1 = self.tensor_to_np(point1)
        p2 = self.tensor_to_np(point2)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=size, height=length)
        
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
            
        cylinder.rotate(rotation_matrix, center=np.array([0, 0, 0]))
        midpoint = (p1 + p2) / 2
        cylinder.translate(midpoint)
        cylinder.paint_uniform_color(color)
        self.geometries.append(cylinder)

    def draw_arrow(self, point1: torch.Tensor, point2: torch.Tensor, size: float = 0.05, color: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        p1 = self.tensor_to_np(point1)
        p2 = self.tensor_to_np(point2)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
            
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=size, cone_radius=size*2, cylinder_height=length*0.9, cone_height=length*0.1)
        
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
            
        arrow.rotate(rotation_matrix, center=np.array([0, 0, 0]))
        arrow.translate(p1)
        arrow.paint_uniform_color(color)
        self.geometries.append(arrow)

if __name__ == "__main__":
    p1 = torch.tensor([0.0, 0.0, 1.0])
    p2 = torch.tensor([1.0, 1.0, 2.0])
    p3 = torch.tensor([2.0, 4.0, 3.0])
    p4 = torch.tensor([3.0, 9.0, 4.0])

    plotter = Plotter()
    plotter.draw_axis(size=1.0)
    plotter.draw_point(p1)
    plotter.draw_point(p2)
    plotter.draw_point(p3)
    plotter.draw_arrow(p1, p2, color=(random.random(),random.random(),random.random()))
    plotter.draw_line(p2, p3, color=(random.random(),random.random(),random.random()))
    plotter.show()
