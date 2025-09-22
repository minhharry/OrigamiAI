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
import time
import numpy as np
import open3d as o3d
from typing import Callable

def show_origami_object_open3d_new(
    origami,
    solverstep: Callable[[object],Optional[OrigamiObject] ],
    fps: int = 30,
    show_points: bool = True,
    show_faces: bool = True,
    show_forces: bool = True,
    show_normals: bool = True,
    force_scale: float = 0.1,
    point_scale: float = 10.0,
    normal_scale: float = 0.2,
    window_name: str = "Origami Viewer"
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    # --- render options ---
    render_option = vis.get_render_option()
    render_option.point_size = point_scale

    def _pts_to_np(points):
        return np.array([[p.position[0].item(),
                          p.position[1].item(),
                          p.position[2].item()] for p in points],
                        dtype=np.float64)

    pts = _pts_to_np(origami.listPoints)

    # --- point cloud ---
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)
    if show_points:
        point_cloud.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([0.6, 0.0, 0.6], dtype=np.float64), (len(pts), 1))
        )
        vis.add_geometry(point_cloud)

    # --- force vectors ---
    force_lines = None
    if show_forces:
        force_lines = o3d.geometry.LineSet()
        vis.add_geometry(force_lines)

    # --- normals as arrows ---
    normal_lines = None
    if show_normals:
        normal_lines = o3d.geometry.LineSet()
        vis.add_geometry(normal_lines)

    # --- line setup ---
    line_color_map = {
        LineType.MOUNTAIN: np.array([1.0, 0.0, 0.0], dtype=np.float64),
        LineType.VALLEY:   np.array([0.0, 0.0, 1.0], dtype=np.float64),
        LineType.BORDER:   np.array([0.0, 0.0, 0.0], dtype=np.float64),
        LineType.FACET:    np.array([0.5, 0.5, 0.5], dtype=np.float64),
    }
    active_line_sets: dict[LineType, o3d.geometry.LineSet] = {}

    # --- mesh setup ---
    mesh = None
    if show_faces and origami.listFaces:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts)
        triangles = np.array(
            [[f.point1Index, f.point2Index, f.point3Index] for f in origami.listFaces],
            dtype=np.int32
        )
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_triangle_normals(normalized=True)
        mesh.compute_vertex_normals(normalized=True)
        vis.add_geometry(mesh)

        # --- auto camera adjust for [-1,1] ---
        view_ctl = vis.get_view_control()
        bounds = mesh.get_axis_aligned_bounding_box()
        view_ctl.set_lookat(bounds.get_center())
        view_ctl.set_up([0, 1, 0])
        view_ctl.set_front([0, 0, -1])
        view_ctl.set_zoom(0.8)

    target_dt = 1.0 / max(1, fps)
    last_time = time.perf_counter()

    try:
        while True:
            if not vis.poll_events():
                break

            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed < target_dt:
                time.sleep(max(0.0, target_dt - elapsed))
            last_time = time.perf_counter()

            # --- run solver ---
            solverstep(origami)

            # --- update points ---
            pts = _pts_to_np(origami.listPoints)
            if show_points:
                point_cloud.points = o3d.utility.Vector3dVector(pts)
                if len(pts) > 0:
                    point_cloud.colors = o3d.utility.Vector3dVector(
                        np.tile(np.array([0.6, 0.0, 0.6], dtype=np.float64), (len(pts), 1))
                    )
                vis.update_geometry(point_cloud)

            # --- update forces ---
            if show_forces and force_lines is not None:
                force_pts, force_edges, force_colors = [], [], []
                for i, p in enumerate(origami.listPoints):
                    ppos = np.array([p.position[0].item(),
                                     p.position[1].item(),
                                     p.position[2].item()])
                    fvec = np.array([p.force[0].item(),
                                     p.force[1].item(),
                                     p.force[2].item()]) * force_scale
                    if np.linalg.norm(fvec) > 1e-9:
                        start_idx = len(force_pts)
                        force_pts.append(ppos)
                        force_pts.append(ppos + fvec)
                        force_edges.append([start_idx, start_idx + 1])
                        force_colors.append([0.0, 1.0, 0.0])  # green

                if force_pts:
                    force_lines.points = o3d.utility.Vector3dVector(np.array(force_pts))
                    force_lines.lines = o3d.utility.Vector2iVector(np.array(force_edges, dtype=np.int32))
                    force_lines.colors = o3d.utility.Vector3dVector(np.array(force_colors))
                    vis.update_geometry(force_lines)

            # --- update lines ---
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
                        ls.colors = o3d.utility.Vector3dVector(
                            np.tile(line_color_map[lt], (len(lines_idx), 1))
                        )
                        vis.add_geometry(ls)
                        active_line_sets[lt] = ls
                    else:
                        ls = active_line_sets[lt]
                        ls.points = o3d.utility.Vector3dVector(pts)
                        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines_idx, dtype=np.int32))
                        ls.colors = o3d.utility.Vector3dVector(
                            np.tile(line_color_map[lt], (len(lines_idx), 1))
                        )
                        vis.update_geometry(ls)

            # --- update faces + normals ---
            if show_faces:
                if origami.listFaces:
                    if mesh is None:
                        mesh = o3d.geometry.TriangleMesh()
                        vis.add_geometry(mesh)
                    mesh.vertices = o3d.utility.Vector3dVector(pts)
                    tris = np.array(
                        [[f.point1Index, f.point2Index, f.point3Index] for f in origami.listFaces],
                        dtype=np.int32
                    )
                    mesh.triangles = o3d.utility.Vector3iVector(tris)
                    mesh.compute_triangle_normals(normalized=True)
                    mesh.compute_vertex_normals(normalized=True)
                    vis.update_geometry(mesh)

                    if show_normals and normal_lines is not None:
                        n_pts, n_edges, n_colors = [], [], []
                        for f in origami.listFaces:
                            tri = np.array([pts[f.point1Index],
                                            pts[f.point2Index],
                                            pts[f.point3Index]])
                            centroid = tri.mean(axis=0)
                            # use mesh normals
                            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                            if np.linalg.norm(normal) > 1e-9:
                                normal /= np.linalg.norm(normal)
                            start_idx = len(n_pts)
                            n_pts.append(centroid)
                            n_pts.append(centroid + normal * normal_scale)
                            n_edges.append([start_idx, start_idx + 1])
                            n_colors.append([1.0, 0.5, 0.0])  # orange for normals
                        if n_pts:
                            normal_lines.points = o3d.utility.Vector3dVector(np.array(n_pts))
                            normal_lines.lines = o3d.utility.Vector2iVector(np.array(n_edges, dtype=np.int32))
                            normal_lines.colors = o3d.utility.Vector3dVector(np.array(n_colors))
                            vis.update_geometry(normal_lines)

                else:
                    if mesh is not None:
                        vis.remove_geometry(mesh)
                        mesh = None

            vis.update_renderer()
    finally:
        vis.destroy_window()
