import math
import numpy as np
from numba import jit

@jit(nopython=True)
def linesCollided(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            return True
        return False
    except Exception:
        return False


@jit(nopython=True)
def getCollisionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        if 0 <= uA <= 1 and 0 <= uB <= 1:
            intersectionX = x1 + (uA * (x2 - x1))
            intersectionY = y1 + (uA * (y2 - y1))
            return intersectionX, intersectionY
        return None
    except Exception:
        return None

@jit(nopython=True)
def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

@jit(nopython=True)
def normalise(data, sensor_size):
    return data / sensor_size

@jit(nopython=True)
def rotate_point(old_x, old_y, cx, cy, angle):
    tempX = old_x - cx
    tempY = old_y - cy

    # now apply rotation
    rotatedX = tempX * math.cos(math.radians(-angle)) - tempY * math.sin(math.radians(-angle))
    rotatedY = tempX * math.sin(math.radians(-angle)) + tempY * math.cos(math.radians(-angle))

    x = rotatedX + cx
    y = rotatedY + cy

    return x, y
