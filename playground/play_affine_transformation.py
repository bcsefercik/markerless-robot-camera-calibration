import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transformation import get_affine_transformation, select_closest_points_to_line

# input data
# inp = [[1, 1, 2], [2, 3, 0], [3, 2, -2], [-2, 2, 3]]  # <- points
# out = [[0, 2, 1], [1, 2, 2], [-2, -1, 6], [4, 1, -3]]  # <- mapped to
inp = [[1, 1, 2], [2, 3, 0], [3, 2, -2], [-2, 2, 3]]  # <- points
out = [[0, 2, 1], [1, 2, 2], [-2, -1, 6], [4, 1, -3]]  # <- mapped to
# calculations

A, t = get_affine_transformation(inp, out)
# output
print("Affine transformation matrix:\n", A)
print("Affine transformation translation vector:\n", t)
# unittests
print("TESTING:")
for p, P in zip(np.array(inp), np.array(out)):
    image_p = np.dot(A, p) + t
    result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
    print(p, " mapped to: ", image_p, " ; expected: ", P, result)
