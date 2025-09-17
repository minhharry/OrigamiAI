import time
import numpy as np
import open3d as o3d
from typing import Callable, Optional
from object.origami_object import OrigamiObject, LineType
import torch

def _pts_to_np(listPoints):
    if len(listPoints) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.vstack([p.position.detach().cpu().numpy().astype(np.float64) for p in listPoints])
    return arr

def show_origami_object_open3d(
    origami: OrigamiObject,
    modifystep: Callable[[OrigamiObject], Optional[OrigamiObject]],
    fps: int = 30,
    show_points: bool = True,
    show_faces: bool = True,
    window_name: str = "Origami Viewer"
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    pts = _pts_to_np(origami.listPoints)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    if show_points:
        point_cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.6, 0.0, 0.6], dtype=np.float64), (len(pts), 1)))
        vis.add_geometry(point_cloud)

    line_color_map = {
        LineType.MOUNTAIN: np.array([1.0, 0.0, 0.0], dtype=np.float64),
        LineType.VALLEY:   np.array([0.0, 0.0, 1.0], dtype=np.float64),
        LineType.BORDER:   np.array([0.0, 0.0, 0.0], dtype=np.float64),
        LineType.FACET:    np.array([0.5, 0.5, 0.5], dtype=np.float64),
    }

    active_line_sets: dict[LineType, o3d.geometry.LineSet] = {}
    mesh = None
    if show_faces and origami.listFaces:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts)
        triangles = np.array([[f.point1Index, f.point2Index, f.point3Index] for f in origami.listFaces], dtype=np.int32)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    target_dt = 1.0 / max(1, fps)
    last_time = time.perf_counter()

    try:
        while True:
            keep = vis.poll_events()
            if keep is False:
                break

            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed < target_dt:
                time.sleep(max(0.0, target_dt - elapsed))
            last_time = time.perf_counter()

            res = modifystep(origami)
            if isinstance(res, OrigamiObject):
                origami = res

            pts = _pts_to_np(origami.listPoints)
            if show_points:
                point_cloud.points = o3d.utility.Vector3dVector(pts)
                if len(pts) > 0:
                    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.6, 0.0, 0.6], dtype=np.float64), (len(pts), 1)))
                vis.update_geometry(point_cloud)

            for lt in LineType:
                lines_idx = [[ln.p1Index, ln.p2Index] for ln in origami.listLines if ln.lineType == lt]
                if len(lines_idx) == 0:
                    if lt in active_line_sets:
                        vis.remove_geometry(active_line_sets[lt])
                        del active_line_sets[lt]
                else:
                    if lt not in active_line_sets:
                        ls = o3d.geometry.LineSet()
                        ls.points = o3d.utility.Vector3dVector(pts)
                        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines_idx, dtype=np.int32))
                        colors = np.tile(line_color_map[lt], (len(lines_idx), 1))
                        ls.colors = o3d.utility.Vector3dVector(colors)
                        vis.add_geometry(ls)
                        active_line_sets[lt] = ls
                    else:
                        ls = active_line_sets[lt]
                        ls.points = o3d.utility.Vector3dVector(pts)
                        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines_idx, dtype=np.int32))
                        colors = np.tile(line_color_map[lt], (len(lines_idx), 1))
                        ls.colors = o3d.utility.Vector3dVector(colors)
                        vis.update_geometry(ls)

            if show_faces:
                if origami.listFaces:
                    if mesh is None:
                        mesh = o3d.geometry.TriangleMesh()
                        vis.add_geometry(mesh)
                    mesh.vertices = o3d.utility.Vector3dVector(pts)
                    tris = np.array([[f.point1Index, f.point2Index, f.point3Index] for f in origami.listFaces], dtype=np.int32)
                    mesh.triangles = o3d.utility.Vector3iVector(tris)
                    mesh.compute_triangle_normals()
                    mesh.compute_vertex_normals()
                    vis.update_geometry(mesh)
                else:
                    if mesh is not None:
                        vis.remove_geometry(mesh)
                        mesh = None

            vis.update_renderer()
    finally:
        vis.destroy_window()
