import lxml
from object.origami_object import Point, Line, LineType
from utils.snap_to_grid_svg import find_max_min
IMAGE_PATH = "assets/flappingBirdSnapped.svg"

NAMESPACE = '{http://www.w3.org/2000/svg}'
LINE_TAG = NAMESPACE + 'line'
RECT_TAG = NAMESPACE + 'rect'

def is_point_exist(point: Point, listPoints: list[Point], pointMergeTolerance: float = 3.0) -> bool:
    return False

def find_point_index(point: Point, listPoints: list[Point], pointMergeTolerance: float = 3.0) -> int:
    return -1

def create_points(root) -> list[Point]:
    return []

def create_lines(root) -> list[Line]:
    return []


def get_points_line_from_svg(svg_file_path: str) -> tuple[list[Point], list[Line]]:
    root = lxml.etree.parse(svg_file_path).getroot()
    listPoints = create_points(root)
    listLines = create_lines(root)
    return listPoints, listLines


if __name__ == "__main__":
    listPoints, listLines = get_points_line_from_svg(IMAGE_PATH)