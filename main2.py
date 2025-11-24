from utils.get_points_line_from_svg import get_points_line_from_svg, triangulate_all
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from physic_engine.solver2 import OrigamiObjectMatrix
from ptu.ptu import gen_ptu
from ptu.ptu_board import gen_ptu_board
from visualization.show_origami_object import show_origami_object_2d_new
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from visualization.animate import show_origami_object_open3d_obj
import random

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

def convert_to_obj(matrix: OrigamiObjectMatrix):    
    obj = OrigamiObject([],[],[])
    for i in range(len(matrix.faces)):
        line = Line(int(matrix.faces[i][2]),int(matrix.faces[i][3]),LineType.MOUNTAIN,matrix.target_theta[i])
        line.currentTheta = line.lastTheta = matrix.theta[i]
        obj.listLines.append(line)
    for m_point in matrix.points:
        point = Point(float(m_point[0]),float(m_point[1]),float(m_point[2])) 
        obj.listPoints.append(point)
    # for m_face in matrix.faces:
    #     face = Face(int(m_face[0]),int(m_face[1]),int(m_face[2]))
    #     obj.listFaces.append(face)

    return obj


random.seed(4) #4


for i in range(1,10):
    listPoints, listLines = gen_ptu_board(np.pi/i-0.01,10,0.5,True) # list[Point], list[Line], Line: {p1: Point, p2: Point, targetTheta: float}
    listLines_ = []
    # for i in range(len(listLines)):
    #     print(listLines[i].p1,listLines[i].p2)

    for i in range(len(listLines)):
        p1_index = listPoints.index(listLines[i].p1)
        p2_index = listPoints.index(listLines[i].p2)
        targetTheta = torch.tensor(listLines[i].targetTheta) if listLines[i].targetTheta != -999 else torch.tensor(0.0)
        lineType = LineType.VALLEY if targetTheta > 0 else LineType.MOUNTAIN
        if listLines[i].targetTheta == -999: lineType = LineType.BORDER
        listLines_.append(Line(p1_index,p2_index,lineType,targetTheta))
    listPoints = [Point(x.position[0],x.position[2],x.position[1]) for x in listPoints]
    
    listFaces = get_faces_from_points_lines(listPoints, listLines_)
    o = OrigamiObject(listPoints, listLines_, listFaces)
    # show_origami_object_2d_new(o,True,True)
    triangulate_all(listPoints,listLines_)
    print("OK")
    # triangulate_all(listPoints,listLines_)
    # triangulate_all(listPoints,listLines_)
    listFaces = get_faces_from_points_lines(listPoints, listLines_)
    

    inputdict = convert_to_matrix(listPoints, listLines_, listFaces)
    ori = OrigamiObjectMatrix(inputdict["points"]*5,
                            inputdict["lines"],
                            inputdict["faces"],
                            inputdict["target_thetas"],
                            )

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

    import time
    import torch
    start_time = time.time()
    for i in range(10000):
        ori.step()
        # if VISUALIZE and i % 100 == 0:
        #     # obj = convert_to_obj(ori)
        #     # print(obj.listLines)
        #     # print(obj.listPoints)
        #     # show_origami_object_open3d_obj(obj)
        #     points = ori.points.cpu().numpy()
        #     lines = ori.lines.cpu().numpy()
        #     ax.clear()
        #     for line in lines:
        #         p1 = points[line[0]]
        #         p2 = points[line[1]]
        #         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-')

        #     # Draw points (vertices)
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=20) # type: ignore

        #     # Draw faces
        #     face1_indices = ori.faces[0, 3, 4].cpu().numpy()
        #     face2_indices = ori.faces[1, 3, 4].cpu().numpy()

        #     # Face 1 (pink)
        #     face1_points = points[face1_indices]
        #     ax.plot_trisurf(face1_points[:, 0], face1_points[:, 1], face1_points[:, 2], color='pink', alpha=0.5)

        #     # Face 2 (white)
        #     face2_points = points[face2_indices]
        #     ax.plot_trisurf(face2_points[:, 0], face2_points[:, 1], face2_points[:, 2], color='white', alpha=0.5)

        #     ax.set_xlim([-1, 19])
        #     ax.set_ylim([-10, 10])
        #     ax.set_zlim([-1, 19])
        #     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        #     ax.set_title(f'(Frame {i})')
            
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()
        #     # time.sleep(0.1)
        #     if not keep_running:
        #         break

        # Assuming 'fig', 'ax', 'ori', 'i', 'keep_running', 'VISUALIZE', and 'start_time' are defined elsewhere

        if VISUALIZE and i % 100 == 0:
            # Clear the previous plot
            ax.clear()

            # Get points and lines
            points = ori.points.cpu().numpy()
            lines = ori.lines.cpu().numpy()

            # Draw lines (edges)
            for line in lines:
                p1 = points[line[0]]
                p2 = points[line[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-')

            # Draw points (vertices)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=20) # type: ignore

            # --- CORRECTED SECTION: Draw faces ---
            # According to the structure, ori.faces[0] contains the definitions for two adjacent faces.

            # Face 1 (pink) is created by vertices at indices ori.faces[0,0], ori.faces[0,2], ori.faces[0,3]
            for face in ori.faces:
                face1_indices = torch.tensor([
                    face[0], face[2], face[3]
                ]).cpu().numpy()

                # Face 2 (white) is created by vertices at indices ori.faces[0,1], ori.faces[0,2], ori.faces[0,3]
                face2_indices = torch.tensor([
                    face[1], face[2], face[3]
                ]).cpu().numpy()

                # Get the 3D coordinates for each face's vertices
                face1_points = points[face1_indices]
                face2_points = points[face2_indices]

                # Draw Face 1 (pink)
                ax.plot_trisurf(face1_points[:, 0], face1_points[:, 1], face1_points[:, 2], color='pink', alpha=0.5)

                # Draw Face 2 (white)
                ax.plot_trisurf(face2_points[:, 0], face2_points[:, 1], face2_points[:, 2], color='white', alpha=0.5)
                # --- END OF CORRECTED SECTION ---

            # Set plot limits and labels
            ax.set_xlim([-1, 19])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-1, 19])
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(f'(Frame {i})')
            
            # Update the plot window
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if not keep_running:
                break

# dd('Run time: ', time.time() - start_time)
dd('Run time: ', time.time() - start_time)


