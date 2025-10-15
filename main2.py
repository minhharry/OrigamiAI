from utils.get_points_line_from_svg import get_points_line_from_svg
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from physic_engine.solver2 import OrigamiObjectMatrix
import matplotlib.pyplot as plt
import torch
import time

IMAGE_PATH = "assets/M.svg"
listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
listFaces = get_faces_from_points_lines(listPoints, listLines)

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

        if idx1 != face2.point1Index or idx2 != face2.point1Index:
            tmp.append(face2.point1Index)
        elif idx1 != face2.point2Index or idx2 != face2.point2Index:
            tmp.append(face2.point2Index)
        elif idx1 != face2.point3Index or idx2 != face2.point3Index:
            tmp.append(face2.point3Index)
        
        if idx1 != face1.point1Index or idx2 != face1.point1Index:
            tmp.append(face1.point1Index)
        elif idx1 != face1.point2Index or idx2 != face1.point2Index:
            tmp.append(face1.point2Index)
        elif idx1 != face1.point3Index or idx2 != face1.point3Index:
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


inputdict = convert_to_matrix(listPoints, listLines, listFaces)
ori = OrigamiObjectMatrix(inputdict["points"], inputdict["lines"], inputdict["faces"], inputdict["target_thetas"])

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
dt = 1.0/60.0
for i in range(10000):
    ori.step(dt)
    if VISUALIZE and i % 1 == 0:
        points = ori.points.cpu().numpy()
        lines = ori.lines.cpu().numpy()
        ax.clear()
        for line in lines:
            p1 = points[line[0]]
            p2 = points[line[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-')

        # Draw points (vertices)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=20) # type: ignore

        ax.set_xlim([-100, 1300])
        ax.set_ylim([-700, 700])
        ax.set_zlim([-100, 1300])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'(Frame {i})')
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        # time.sleep(0.1)
        if not keep_running:
            break


dd('Run time: ', time.time() - start_time)

