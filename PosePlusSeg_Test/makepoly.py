import numpy as np
from imantics import Polygons, Mask

# This can be any array
array = np.ones((100, 100))

polygons = Mask(array).polygons()

print(polygons.points)
print(polygons.segmentation)