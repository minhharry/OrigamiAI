from object.origami_object import Point, Line, LineType, OrigamiObject, Face
# from utils.get_points_line_from_svg import find_polygons
import torch

def find_polygons(listPoints: list[Point], listLines: list[Line]) -> list[list[int]]:
    # Xây graph kề
    graph = [[] for _ in range(len(listPoints))]
    edges = set()
    for line in listLines:
        u, v = line.p1Index, line.p2Index
        graph[u].append(v)
        graph[v].append(u)
        edges.add((min(u, v), max(u, v)))

    triangles = set()

    # Duyệt từng cạnh và tìm tam giác
    for u, v in edges:
        # giao các điểm kề của u và v
        neighbors_u = set(graph[u])
        neighbors_v = set(graph[v])
        common = neighbors_u.intersection(neighbors_v)
        for w in common:
            tri = tuple(sorted([u, v, w]))
            triangles.add(tri)

    # Convert sang list
    polygons = [list(tri) for tri in triangles]
    return polygons

def triangulate_polygon_to_faces(polygon: list[int]) -> list[Face]:
    faces = []
    if len(polygon) < 3:
        return faces

    p0 = polygon[0]
    for i in range(1, len(polygon) - 1):
        faces.append(Face(p0, polygon[i], polygon[i + 1]))
    return faces

def get_faces_from_points_lines(listPoints, listLines) -> list[Face]:
    polygons = find_polygons(listPoints, listLines)
    all_faces = []
    for poly in polygons:
        faces = triangulate_polygon_to_faces(poly)
        all_faces.extend(faces)
    return all_faces
