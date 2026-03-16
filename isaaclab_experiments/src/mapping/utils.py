import math
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from typing import cast

def bresenham(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def compute_dist(a,b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def distance_transform_edt(mask):
    return cast(np.ndarray, edt(mask))