import matplotlib.pyplot as plt
from object.origami_object import OrigamiObject, LineType

def show_origami_object(origami: OrigamiObject, show_points: bool = True) -> None:
    """
    Visualizes an OrigamiObject in 3D using matplotlib.
    
    Args:
        origami (OrigamiObject): The object to visualize.
        show_points (bool): Whether to plot individual points.
    """
    fig = plt.figure()
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
        ax.scatter(xs, ys, zs, color='purple', s=50, label='Points')

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

    plt.show()

def show_origami_object_2d(origami: OrigamiObject, show_points: bool = True) -> None:
    """
    Visualizes an OrigamiObject in a 2D plane (X-Z) using matplotlib, ignoring the Y-axis.

    Args:
        origami (OrigamiObject): The object to visualize.
        show_points (bool): Whether to plot individual points.
    """
    fig, ax = plt.subplots()

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