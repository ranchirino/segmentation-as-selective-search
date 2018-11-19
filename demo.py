import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import data
from imageio import imread
from datetime import datetime

from selective_search import selective_search

# img = data.coffee()
# img = data.astronaut()
# img = data.chelsea()
img = imread("image1.jpg")
t1 = datetime.now()
segment_mask, R = selective_search(img, scale=150, sim_threshold=0.65)
t2 = datetime.now()
print(t2-t1)


fig = plt.figure()
ax1 = plt.subplot(121)
plt.imshow(img)
for r in R:
    rect = Rectangle((r['x_min'], r['y_min']), r['width'], r['height'], fill=False, color='red', linewidth=1.5)
    ax1.add_patch(rect)
ax2 = plt.subplot(122)
plt.imshow(segment_mask)
plt.show()
