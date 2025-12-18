from visualization.show_origami_object import show_origami_object, show_origami_object_2d, show_origami_object_2d_new, show_faces_2d
from object.origami_object import OrigamiObject, Point, Line, Face, LineType
from utils.get_points_line_from_svg import get_points_line_from_svg, triangulate_all
from utils.get_faces_from_points_lines import get_faces_from_points_lines
from physic_engine.solver import solverStep, setDeltaTime
from physic_engine.solver2 import OrigamiObjectMatrix, convert_to_matrix
from visualization.animate_pointcloud import Plotter
from visualization.animate import show_origami_object_open3d
from ptu.ptu import gen_ptu
from ptu.ptu_board import gen_ptu_board
from utils.points_lines_to_svg import points_lines_to_svg
from utils.save_3D_obj import save_obj, save_obj_from_arrays
from utils.loss import chamfer_with_scale_search
from utils.writeText import writeText
from physic_engine.solver3 import OrigamiObjectMatrixJax
import numpy as np
import time
import torch
import random
from datetime import datetime
import trimesh
import os
from tqdm import tqdm
from visualization.tensor_vis import visualize_simulation

def main():
    IMAGE_PATH = "output/output_920251201140220/output.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object(o)
    show_origami_object_2d(o,True)
    for face in o.listFaces:
        print(face)
    show_faces_2d(o)

def main2():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    print("points:", len(o.listPoints))
    print("lines:", len(o.listLines))
    print("faces:", len(o.listFaces))
    num_iterations = 100
    show_origami_object(o)
    for i in range(num_iterations):
        solverStep(o)
    show_origami_object(o)
    for i in range(num_iterations):
        solverStep(o)
    show_origami_object(o)

def main_ptu():
    random.seed(2) #4, 6,7
    print("main_ptu")
    TARGET_SHAPE_OBJ = "target/bat.obj"
    target_mesh = trimesh.load(TARGET_SHAPE_OBJ, force='mesh')
    num_points = 10000
    target_cloud_points = target_mesh.sample(num_points)
    print("target_cloud_points ==:",target_cloud_points.shape)
    best_loss = 10000
    best_id = 0
    src = f"output"
    id = str(best_id)+datetime.now().strftime("%Y%m%d%H%M%S")
    total_iters = 50*3*7
    start_time = time.time()
    target_folder = "output/output_n"
    if os.path.exists(src):
        os.rename(src,f"output_backup_{id}")
    
    total_n = 7
    total_pi = 3
    total_try = 100

    for i in tqdm(range(total_n*total_pi*total_try), desc="PTU Search", total=total_n*total_pi*total_try):
        iter_start = time.time()
        run_id = f"{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            # id = str(i)+datetime.now().strftime("%Y%m%d%H%M%S")
                    n = i//total_try//total_pi + 8
                    pi = i // total_try % total_pi + 1
                    target_folder = f"output_x_y/output_n_{n}/output_1_pi_{pi}/"
    # for k_pi in range(1, 4):
    #     target_folder_ = f"output/output_k_n_{k_pi}"
    #     for n in range(8, 16):
    #         target_folder = target_folder_+"/"+"n_"+str(n)+"/"
    #         for i in tqdm(range(0, 50), desc="🔁 PTU Search", total=total_iters):
    #             iter_start = time.time()
    #             run_id = f"{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    #             try:
                    # print("i:",i)
                    id = str(i)+datetime.now().strftime("%Y%m%d%H%M%S")
                    listPoints, listLines = gen_ptu_board(np.pi/pi-0.01,n,0.5,id,target_folder,False) # list[Point], list[Line], Line: {p1: Point, p2: Point, targetTheta: float}
                    # print("listPoints:",len(listPoints))
                    listLines_ = []

                    for i in range(len(listLines)):
                        p1_index = listPoints.index(listLines[i].p1)
                        p2_index = listPoints.index(listLines[i].p2)
                        targetTheta = torch.tensor(listLines[i].targetTheta) if listLines[i].targetTheta != -999 else torch.tensor(0.0)
                        lineType = LineType.VALLEY if targetTheta > 0 else LineType.MOUNTAIN
                        if listLines[i].targetTheta == -999: lineType = LineType.BORDER
                        listLines_.append(Line(p1_index,p2_index,lineType,targetTheta))
                    listPoints = [Point(x.position[0],x.position[2],x.position[1]) for x in listPoints]
                    
                    # listFaces = get_faces_from_points_lines(listPoints, listLines_)
                    # show_origami_object_2d_new(o,True,True)
                    triangulate_all(listPoints,listLines_)
                    listFaces = get_faces_from_points_lines(listPoints, listLines_)
                    points_lines_to_svg(listPoints,listLines_,100,f"output.svg",target_folder+f"output_{id}")

                    o = OrigamiObject(listPoints, listLines_, listFaces)
                    # show_origami_object_2d_new(o,True,True)
                    # show_origami_object_open3d(o,solverStep,30,True,True,True,True,2)
                    # print("start solver")
                    for i in range(3000):
                        solverStep(o)
                    # show_origami_object_open3d(o,solverStep,30,True,True,True,True,2)
                    # print("save obj")
                    save_obj(o.listPoints,o.listLines,o.listFaces,f"output.obj",target_folder+f"output_{id}")
                    obj = trimesh.load(target_folder+f"output_{id}/output.obj")
                    obj_cloud_points = obj.sample(num_points)
                    # print("load obj:", obj_cloud_points.shape)
                    loss_value, scale = chamfer_with_scale_search(
                        target_cloud_points,
                        obj_cloud_points
                    )
                    loss_value = loss_value.item()
                    improved = False
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_id = run_id
                        improved = True
                
                    writeText(f"# Chamfer loss: {loss_value} best scale is: {scale}\n", f"output.txt", f"{target_folder}output_{id}")
                    print(f"Chamfer loss: {loss_value} with scale: {scale}")
                    writeText(f"# Best Chamfer loss: {best_loss} with id: {best_id}\n", f"output_best.txt", f"{target_folder}")
                    elapsed = time.time() - iter_start
                    status = "🔥 NEW BEST" if improved else "ok"
                    tqdm.write(
                        f"[{i}] loss={loss_value:.6f} | best={best_loss:.6f} "
                        f"| {elapsed:.1f}s | {status}"
                    )
        except Exception as e:
            print(e)
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Finished in {total_time:.1f} minutes")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"🆔 Best id: {best_id}")

def show_full():
    IMAGE_PATH = "assets/M.svg"
    # listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listPoints, listLines = gen_ptu(np.pi/2,4,5)
    listLines_ = []
    for i in range(len(listLines)):
        p1_index = listPoints.index(listLines[i].p1)
        p1_index = listPoints.index(listLines[i].p2)
        targetTheta = torch.tensor(listLines[i].targetTheta)
        listLines_.append(Line(p1_index,p1_index,LineType.MOUNTAIN,targetTheta))
    listPoints = [Point(x.position[0],x.position[1],x.position[2]) for x in listPoints]
    listFaces = get_faces_from_points_lines(listPoints, listLines_)
    o = OrigamiObject(listPoints, listLines_, listFaces)
    setDeltaTime(o)
    # o.listPoints[0].is_fixed = True
    show_origami_object_open3d(o,solverStep,30,True,True,True,True,2)

def ptu_new_solver():
    random.seed(2)
    print("ptu_new_solver")

    TARGET_SHAPE_OBJ = "target/bat.obj"
    target_mesh = trimesh.load(TARGET_SHAPE_OBJ, force='mesh')
    num_points = 5000
    target_cloud_points = target_mesh.sample(num_points)

    print("target_cloud_points ==:",target_cloud_points.shape)
    best_loss = 5000
    best_id = 0

    src = f"output"
    id = str(best_id)+datetime.now().strftime("%Y%m%d%H%M%S")
    total_iters = 18
    start_time = time.time()
    # if os.path.exists(src):
    #     os.rename(src,f"output_backup_{id}")
    total_n = 5
    total_pi = 2
    total_try = 100
    top_n = 20
    list_top_n = []
    for i in tqdm(range(total_n*total_pi*total_try), desc="PTU Search", total=total_n*total_pi*total_try):
        iter_start = time.time()
        run_id = f"{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            id = str(i)+datetime.now().strftime("%Y%m%d%H%M%S")
            n = i//total_try//total_pi*3 + 12
            pi = i // total_try % total_pi + 1 
            target_folder = f"output_new_bat/y_y/output_n_{n}/output_1_pi_{pi}/"
            listPoints, listLines = gen_ptu_board(np.pi/pi-0.01,n,0.5,id,target_folder,False) # list[Point], list[Line], Line: {p1: Point, p2: Point, targetTheta: float}
            listLines_ = []
            for i in range(len(listLines)):
                p1_index = listPoints.index(listLines[i].p1)
                p2_index = listPoints.index(listLines[i].p2)
                targetTheta = torch.tensor(listLines[i].targetTheta) if listLines[i].targetTheta != -999 else torch.tensor(0.0)
                lineType = LineType.VALLEY if targetTheta > 0 else LineType.MOUNTAIN
                if listLines[i].targetTheta == -999: lineType = LineType.BORDER
                listLines_.append(Line(p1_index,p2_index,lineType,targetTheta))
            listPoints = [Point(x.position[0],x.position[2],x.position[1]) for x in listPoints]
            
            triangulate_all(listPoints,listLines_)
            listFaces = get_faces_from_points_lines(listPoints, listLines_)
            points_lines_to_svg(listPoints,listLines_,100,f"output.svg",f"{target_folder}output_{id}")
            # print("OK")
            converted_data = convert_to_matrix(listPoints, listLines_, listFaces)
            points_tensor = converted_data["points"]
            lines_tensor = converted_data["lines"]
            faces_tensor = converted_data["faces"]
            target_theta_tensor = converted_data["target_thetas"]
            # print("converted success")
            o = OrigamiObjectMatrixJax(
                points=points_tensor,
                lines=lines_tensor,
                faces=faces_tensor,
                target_theta=target_theta_tensor,
                mass=1.0,
                ea=20.0,
                k_crease=0.7,
                damping=0.45,
                fold_percent=0.99,
                dt=-1.0,
                k_facet=50
            )
            # print("run steps")
            points_jax = o.run_steps(num_steps=5000)
            # print(points_jax.squeeze().numpy())
            # visualize_simulation(o,num_steps=3000,run_all_steps=True)
            # print("run success")
            save_obj_from_arrays(points_jax.squeeze().numpy(),o.faces_jax,f"output.obj",f"{target_folder}output_{id}")
            obj = trimesh.load(f"{target_folder}/output_{id}/output.obj")

            obj_cloud_points = obj.sample(num_points)
            loss_value, scale = chamfer_with_scale_search(
                target_cloud_points,
                obj_cloud_points
            )
            loss_value = loss_value.item()
            improved = False
            if loss_value < best_loss:
                best_loss = loss_value
                best_id = run_id
                improved = True
            
            if len(list_top_n) < top_n: 
                list_top_n.append(tuple([loss_value, run_id]))
            elif loss_value < min(x[0] for x in list_top_n):
                list_top_n.append(tuple([loss_value, run_id]))
            list_top_n = sorted(list_top_n, key=lambda x: x[0])
            list_top_n = list_top_n[:top_n]

            writeText(f"# Chamfer loss: {loss_value} best scale is: {scale}\n", f"output.txt", f"{target_folder}output_{id}")
            # print(f"Chamfer loss: {loss_value} with scale: {scale}")
            
            writeText(f"", f"output_best.txt", f"{target_folder}")
            for i in range(len(list_top_n)):
                writeText(f"# Chamfer loss: {list_top_n[i][0]} with id: {list_top_n[i][1]}\n", f"output_best.txt", f"{target_folder}", isAppend=True)
            elapsed = time.time() - iter_start
            status = "NEW BEST" if improved else "ok"
            tqdm.write(
                f"[{i}] loss={loss_value:.6f} | best={best_loss:.6f} "
                f"| {elapsed:.1f}s | {status}"
            )
        except Exception as e:
            print(e)
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Finished in {total_time:.1f} minutes")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print(f"🆔 Best id: {best_id}")

def main4():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object_2d_new(o,True,True)

def show_pointcloud():
    IMAGE_PATH = "assets/flappingBird.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    setDeltaTime(o)
    # o.listPoints[0].is_fixed = True
    plotter = Plotter(o, solverStep)
    plotter.show()

def benchmark():
    IMAGE_PATH = "assets/M.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    start_time = time.time()
    for i in range(1000):
        solverStep(o)
    print("time for 1000 steps of solverStep:", time.time() - start_time)

def show_output():
    IMAGE_PATH = "output_new_bat/y_y/output_n_12/output_1_pi_1/output_1920251218073524/output.svg"
    # IMAGE_PATH = "output_new/output_n_8/output_1_pi_2/output_97520251217115221/output.svg"
    # IMAGE_PATH = "output_new_y_x_y/output_n_6/output_1_pi_2/output_1620251217221603/output.svg"
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)
    listFaces = get_faces_from_points_lines(listPoints, listLines)
    o = OrigamiObject(listPoints, listLines, listFaces)
    show_origami_object_2d_new(o,True,True)
    show_origami_object_open3d(o,solverStep,30)


if __name__ == "__main__":

    a = [1,2,3,4,5,6,7,8,9,10]
    print(a[0:5])
    
    import argparse
    import sys

    # 1. Create the parser
    parser = argparse.ArgumentParser(description="A simple file processor.")

    # 2. Add arguments
    parser.add_argument("function", help="function to call")

    # 3. Parse the arguments
    args = parser.parse_args()

    mapping = {
        'main': main,
        'main2': main2,
        'main_ptu': main_ptu,
        'show_full': show_full,
        # 'show_full2': show_full2,
        'main4': main4,
        'show_pointcloud': show_pointcloud,
        'benchmark': benchmark,
        'show_output': show_output,
        'ptu_new_solver': ptu_new_solver,
    }

    # 4. Use the arguments in your script
    try:
        mapping[args.function]()

    except Exception as e:
        print(f"Error: The function '{args.function}' was not found.")
        print("Err",e)
        sys.exit(1)
