from lxml import etree # type: ignore

IMG_PATH = "assets/airplane_.svg"



def normalize_img(input_path: str, output_path: str) -> None:
    tree = etree.parse(input_path)
    root = tree.getroot()

    min_x, max_x =99999,-99999
    min_y, max_y = 99999, -99999
    def transform(x, y):
        return ((x - min_x) * scale, (y - min_y) * scale)
    ns = {"svg": "http://www.w3.org/2000/svg"}

    crease_group = root.find(".//svg:g[@class='crease']", ns)
    for line in crease_group.findall("svg:line", ns):
        x1, y1 = float(line.get("x1")), float(line.get("y1"))
        x2, y2 = float(line.get("x2")), float(line.get("y2"))
        min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
        min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)

    width, height = 1000, 1000
    dx, dy = max_x - min_x, max_y - min_y
    scale_x = width / dx if dx > 0 else 1
    scale_y = height / dy if dy > 0 else 1
    scale = min(scale_x, scale_y)   
    TARGET = 1000

    # Scale so the largest side fits into TARGET
    if dx >= dy:
        scale = TARGET / dx if dx > 0 else 1
        width = TARGET
        height = dy * scale
    else:
        scale = TARGET / dy if dy > 0 else 1
        width = dx * scale
        height = TARGET

    # Create root SVG with proportional width/height
    svg = etree.Element(
        "svg",
        version="1.1",
        xmlns="http://www.w3.org/2000/svg",
        width=f"{width}px",
        height=f"{height}px"
    )
    etree.SubElement(svg, "rect", {
            "x": "0", "y": "0",
            "width": f"{width}", "height": f"{height}",
            "fill": "#FFFFFF",
            "stroke": "#000000",
            "stroke-width": "10"
    })
    for line in crease_group.findall("svg:line", ns):
        stroke = line.get("stroke")
        if stroke in ["#000000"]: continue
        x1 = float(line.get("x1"))
        y1 = float(line.get("y1"))
        x2 = float(line.get("x2"))
        y2 = float(line.get("y2"))
        x1_,y1_ = transform(x1, y1)
        x2_,y2_ = transform(x2, y2)
        opacity = 1
        if line.get("stroke-opacity"): opacity =  line.get("stroke-opacity")
        elif line.get("opacity"): opacity = line.get("opacity")
        etree.SubElement(svg, "line", {
            "x1": str(x1_), "y1": str(y1_),
            "x2": str(x2_), "y2": str(y2_),
            "stroke": stroke,
            "stroke-width": "10",
            "stroke-opacity": opacity
        })

    with open(output_path, "wb") as f:
        f.write(etree.tostring(svg, pretty_print=True, xml_declaration=True, encoding="utf-8"))


normalize_img(IMG_PATH,"output.svg")