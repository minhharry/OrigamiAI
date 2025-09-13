from lxml import etree

NAMESPACE = '{http://www.w3.org/2000/svg}'
LINE_TAG = NAMESPACE + 'line'
RECT_TAG = NAMESPACE + 'rect'
GRID_SIZE = 32
SCALE = 100

tree = etree.parse('assets/flappingBird.svg')
root = tree.getroot()

def find_max_min(root: etree.Element) -> tuple[float, float, float, float]:
    minx = 100000
    miny = 100000
    maxx = -100000
    maxy = -100000

    for child in root:
        if child.tag == RECT_TAG:
            minx = min(minx, float(child.attrib['x']))
            miny = min(miny, float(child.attrib['y']))
            maxx = max(maxx, float(child.attrib['x']) + float(child.attrib['width']))
            maxy = max(maxy, float(child.attrib['y']) + float(child.attrib['height']))
        if child.tag == LINE_TAG:
            minx = min(minx, float(child.attrib['x1']))
            miny = min(miny, float(child.attrib['y1']))
            maxx = max(maxx, float(child.attrib['x2']))
            maxy = max(maxy, float(child.attrib['y2']))
    return minx, miny, maxx, maxy

# Translate the image to the top left.
minx, miny, maxx, maxy = find_max_min(root)

for child in root:
    if child.tag == RECT_TAG:
        child.attrib['x'] = str(float(child.attrib['x']) - minx)
        child.attrib['y'] = str(float(child.attrib['y']) - miny)
    if child.tag == LINE_TAG:
        child.attrib['x1'] = str(float(child.attrib['x1']) - minx)
        child.attrib['x2'] = str(float(child.attrib['x2']) - minx)
        child.attrib['y1'] = str(float(child.attrib['y1']) - miny)
        child.attrib['y2'] = str(float(child.attrib['y2']) - miny)

# Snap to grid
minx, miny, maxx, maxy = find_max_min(root)
for child in root:
    if child.tag == RECT_TAG:
        child.attrib['x'] = str(round(float(child.attrib['x'])/maxx * GRID_SIZE)*SCALE)
        child.attrib['y'] = str(round(float(child.attrib['y'])/maxy * GRID_SIZE)*SCALE)
        child.attrib['width'] = str(round(float(child.attrib['width'])/maxx * GRID_SIZE)*SCALE)
        child.attrib['height'] = str(round(float(child.attrib['height'])/maxy * GRID_SIZE)*SCALE)
    if child.tag == LINE_TAG:
        child.attrib['x1'] = str(round(float(child.attrib['x1'])/maxx * GRID_SIZE)*SCALE)
        child.attrib['x2'] = str(round(float(child.attrib['x2'])/maxx * GRID_SIZE)*SCALE)
        child.attrib['y1'] = str(round(float(child.attrib['y1'])/maxy * GRID_SIZE)*SCALE)
        child.attrib['y2'] = str(round(float(child.attrib['y2'])/maxy * GRID_SIZE)*SCALE)

root.attrib['viewBox'] = f'0 0 3200 3200'
tree.write('output.svg', encoding='utf-8', xml_declaration=True)
