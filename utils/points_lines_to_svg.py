from lxml import etree # type: ignore
from object.origami_object import Point, Line, LineType
import math
import os
IMG_PATH = "assets/airplane_.svg"



def points_lines_to_svg(points: list[Point], lines: list[Line], scale: float, file_name: str, folder_path: str = "") -> None:

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if scale <= 0:
        scale = 1
    if folder_path == "":
        folder_path = os.path.join(BASE_DIR, "outputSvg")

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file_name)

    min_x = min(p.position[0].item() for p in points)
    max_x = max(p.position[0].item() for p in points)
    min_y = min(p.position[2].item() for p in points)
    max_y = max(p.position[2].item() for p in points)
        
    canvas_size_w = max_x - min_x
    canvas_size_h = max_y - min_y

    def transform(x, y):
        return ( (x - min_x) * scale, (y - min_y) * scale )
    
    svg = etree.Element(
        "svg",
        version="1.1",
        xmlns="http://www.w3.org/2000/svg",
        width=str(float(canvas_size_w)*scale)+"px",
        height=str(float(canvas_size_h)*scale)+"px"
    )
    etree.SubElement(svg, "rect", {
        "x": "0",
        "y": "0",
        "width": str(float(canvas_size_w) * scale),
        "height": str(float(canvas_size_h) * scale),
        "fill": "#FFFFFF"
    })

    for line in lines:
        if line.lineType == LineType.FACET:
            continue 
        if line.lineType == LineType.BORDER:
            color = "#000000"
        elif line.lineType == LineType.MOUNTAIN:
            color = "#FF0000"
        elif line.lineType == LineType.VALLEY:
            color = "#0000FF"
        else:
            color = "#888888"  # fallback
        
        # Compute points
        p1 = points[line.p1Index].position
        p2 = points[line.p2Index].position

        x1, y1 = transform(p1[0].item(), p1[2].item())
        x2, y2 = transform(p2[0].item(), p2[2].item())
        
        # Compute opacity based on targetTheta/pi
        if hasattr(line, "targetTheta") and line.lineType in [LineType.MOUNTAIN,LineType.VALLEY]:
            opacity = min(max(abs(line.targetTheta.item() / math.pi), 0.1), 1.0)
        else:
            opacity = 1.0
        
        etree.SubElement(svg, "line", {
            "x1": str(x1), "y1": str(y1),
            "x2": str(x2), "y2": str(y2),
            "stroke": color,
            "stroke-width": "2",
            "stroke-opacity": str(opacity)
        })
    
    with open(file_path, "wb") as f:
        f.write(etree.tostring(svg, pretty_print=True, xml_declaration=True, encoding="utf-8"))


