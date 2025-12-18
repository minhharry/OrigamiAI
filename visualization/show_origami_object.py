import matplotlib.pyplot as plt
from object.origami_object import OrigamiObject, LineType
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

def set_axes_equal(ax):
    """
    Makes axes of a 3D plot have an equal scale, so that spheres appear as spheres,
    and cubes as cubes.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
def show_origami_object(origami: OrigamiObject, show_points: bool = True) -> None:
    """
    Visualizes an OrigamiObject in 3D using matplotlib.
    
    Args:
        origami (OrigamiObject): The object to visualize.
        show_points (bool): Whether to plot individual points.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for different line types
    line_colors = {
        LineType.MOUNTAIN: 'red',
        LineType.VALLEY: 'blue',
        LineType.BORDER: 'black',
        LineType.FACET: 'gray'
    }

    # Plot lines based on their type
    for line in origami.listLines:
        p1 = origami.listPoints[line.p1Index].position
        p2 = origami.listPoints[line.p2Index].position
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=line_colors.get(line.lineType, 'gray'),
            label=line.lineType.name
        )

    # Plot points if requested
    if show_points:
        points_positions = [point.position for point in origami.listPoints]
        xs = [p[0] for p in points_positions]
        ys = [p[1] for p in points_positions]
        zs = [p[2] for p in points_positions]
        ax.scatter(xs, ys, zs, color='purple', s=50, label='Points') # type: ignore

    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Origami Object Visualization')
    
    # Create a single legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels)

    set_axes_equal(ax)
    
    plt.show()

def show_origami_object_2d(origami: OrigamiObject, show_points: bool = True) -> None:


    """
    Visualizes an OrigamiObject in a 2D plane (X-Z) using matplotlib, ignoring the Y-axis.

    Args:
        origami (OrigamiObject): The object to visualize.
        show_points (bool): Whether to plot individual points.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors for different line types
    line_colors = {
        LineType.MOUNTAIN: 'red',
        LineType.VALLEY: 'blue',
        LineType.BORDER: 'black',
        LineType.FACET: 'gray'
    }

    # Plot lines based on their type, using x and z coordinates
    for line in origami.listLines:
        p1 = origami.listPoints[line.p1Index].position
        p2 = origami.listPoints[line.p2Index].position
        ax.plot(
            [p1[0], p2[0]],
            [p1[2], p2[2]],
            color=line_colors.get(line.lineType, 'gray'),
            label=line.lineType.name
        )

    # Plot points if requested, using x and z coordinates
    if show_points:
        points_positions = [point.position for point in origami.listPoints]
        xs = [p[0] for p in points_positions]
        zs = [p[2] for p in points_positions]
        ax.scatter(xs, zs, color='purple', s=50, label='Points')

    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Z Axis')
    ax.set_title('Origami Object Visualization (2D, X-Z Plane)')
    ax.grid(True)
    ax.set_aspect('equal', 'box') # Keep aspect ratio equal for better visualization
    
    # Create a single legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels)

    plt.show()

def show_origami_object_2d_new(origami: OrigamiObject, show_points: bool = True, show_lines: bool = True) -> None:
    listPoints, listLines = origami.listPoints, origami.listLines
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, line in enumerate(listLines):
        p1 = listPoints[line.p1Index].position
        p2 = listPoints[line.p2Index].position

        x_values = [p1[0].item(), p2[0].item()]  # x
        y_values = [p1[2].item(), p2[2].item()]  # z (thay cho y)

        if line.lineType.name == "MOUNTAIN":
            color = "red"
            linestyle = "--"
        elif line.lineType.name == "VALLEY":
            color = "blue"
            linestyle = ":"
        elif line.lineType.name == "BORDER":
            color = "black"
            linestyle = "-"
        else:  # FACET
            color = "gray"
            linestyle = "-"

        # Vẽ line
        ax.plot(x_values, y_values, color=color, linestyle=linestyle)
        if not show_lines:
            continue
        # Đặt số ở giữa line
        mid_x = (x_values[0] + x_values[1]) / 2
        mid_y = (y_values[0] + y_values[1]) / 2
        # ax.text(mid_x, mid_y, str(i), color="green", fontsize=8)

    # Vẽ point và đánh số
    if show_points:
        for j, point in enumerate(listPoints):
            x, y = point.position[0].item(), point.position[2].item()
            ax.scatter(x, y, color="black", s=10)  # chấm đen nhỏ
            # ax.text(x + 0.001, y + 0.001, str(j), color="purple", fontsize=8)  # số hơi lệch để dễ nhìn

    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # SVG gốc có trục y lộn ngược
    plt.show()

def show_faces_2d(origami_obj: OrigamiObject):
    """
    Hiển thị tất cả Face (tam giác) trong OrigamiObject bằng matplotlib.
    """
    listPoints = origami_obj.listPoints
    listFaces = origami_obj.listFaces

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, face in enumerate(listFaces):
        # Lấy tọa độ 3 điểm
        xs = [
            listPoints[face.point1Index].position[0].item(),
            listPoints[face.point2Index].position[0].item(),
            listPoints[face.point3Index].position[0].item()
        ]
        ys = [
            listPoints[face.point1Index].position[2].item(),
            listPoints[face.point2Index].position[2].item(),
            listPoints[face.point3Index].position[2].item()
        ]

        # Vẽ tam giác (fill + border)
        ax.fill(xs, ys, alpha=0.3)
        ax.plot(xs + [xs[0]], ys + [ys[0]], color="black")

    # Vẽ point
    for j, point in enumerate(listPoints):
        x, y = point.position[0].item(), point.position[2].item()
        ax.scatter(x, y, color="red", s=15)
        ax.text(x + 0.5, y + 0.5, str(j), color="blue", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
