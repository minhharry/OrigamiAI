from object.origami_object import Point, Line, LineType, OrigamiObject, Face
# from utils.get_points_line_from_svg import find_polygons
import torch

def find_polygons(listPoints: list[Point], listLines: list[Line]) -> list[list[int]]:
    graph = [[] for _ in range(len(listPoints))]
    edges = set()
    try:
        for line in listLines:
            u, v = line.p1Index, line.p2Index
            graph[u].append(v)
            graph[v].append(u)
            edges.add((min(u, v), max(u, v)))
    except Exception as e:
        print(e)
   
    triangles = set()

    for u, v in edges:
        neighbors_u = set(graph[u])
        neighbors_v = set(graph[v])
        common = neighbors_u.intersection(neighbors_v)
        for w in common:
            tri = tuple(sorted([u, v, w]))
            triangles.add(tri)

    # Convert sang list
    polygons = [list(tri) for tri in triangles]
    return polygons
def triangulate_polygon_to_faces(polygon: list[int], listPoints: list[Point]) -> list[Face]:
    faces = []
    if len(polygon) < 3:
        return faces
    p0 = polygon[0]
    for i in range(1, len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]

        A = listPoints[p0].originalPosition
        B = listPoints[p1].originalPosition
        C = listPoints[p2].originalPosition

        cross = (B[0] - A[0]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[0] - A[0])

        if cross < 0:
            faces.append(Face(p0, p1, p2))
        else:
            faces.append(Face(p0, p2, p1))

    return faces


def get_faces_from_points_lines(listPoints, listLines) -> list[Face]:
    polygons = find_polygons(listPoints, listLines)
    all_faces = []
    for poly in polygons:
        faces = triangulate_polygon_to_faces(poly,listPoints)
        all_faces.extend(faces)
    return all_faces
