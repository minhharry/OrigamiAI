from object.origami_object import OrigamiObject
# Solver
# 1. Calculate face normals of all triangular faces in mesh (one face per thread).
# 2. Calculate current fold angle for all edges in mesh (one edge per thread).
# 3. Calculate coefficients of Equations 3–6 for all edges in mesh (one edge per
# thread).
# 4. Calculate forces and velocities for all nodes in mesh (one node per thread).
# 5. Calculate positions for all nodes in mesh (one node per thread).

def solver(objectOrigami: OrigamiObject) -> None:
    # Nhận Object vào solver rồi chạy 1 step, sửa objectOrigami
    return