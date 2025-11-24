import polyscope as ps
import numpy as np
import torch


def process_faces(faces_tensor: torch.Tensor):
        faces = faces_tensor.cpu().detach().numpy()
        p0 = faces[:, 0]
        p1 = faces[:, 1]
        p2 = faces[:, 2]
        p3 = faces[:, 3]
        t1_front = np.stack([p2, p0, p3], axis=1)
        t2_front = np.stack([p2, p3, p1], axis=1)
        faces_front = np.vstack([t1_front, t2_front]).astype(np.int32)
        t1_back = np.stack([p2, p3, p0], axis=1)
        t2_back = np.stack([p2, p1, p3], axis=1)
        faces_back = np.vstack([t1_back, t2_back]).astype(np.int32)
        return faces_front, faces_back

def visualize_simulation(ori):
    """
    Visualize simulation using Polyscope
    Args:
        ori: OrigamiObjectMatrix
    """
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    faces_front, faces_back = process_faces(ori.faces)
    initial_points = ori.points.cpu().detach().numpy()
    ps_mesh_front = ps.register_surface_mesh("Front (Blue)", initial_points, faces_front, smooth_shade=True)
    ps_mesh_front.set_color([0.2, 0.5, 1.0])
    ps_mesh_front.set_edge_width(1.0)
    ps_mesh_front.set_back_face_policy("cull")
    ps_mesh_back = ps.register_surface_mesh("Back (Gray)", initial_points, faces_back, smooth_shade=True)
    ps_mesh_back.set_color([0.6, 0.6, 0.6])
    ps_mesh_back.set_edge_width(1.0)
    ps_mesh_back.set_back_face_policy("cull")
    frame_count = 0
    while not ps.window_requests_close():
        if frame_count < 10000:
            ori.step()
            frame_count += 1
        points_np = ori.points.cpu().detach().numpy()
        ps_mesh_front.update_vertex_positions(points_np)
        ps_mesh_back.update_vertex_positions(points_np)
        ps.frame_tick()