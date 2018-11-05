import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import segmentation
from skimage.data import coffee
# import selectivesearch

img = coffee()
# plt.imshow(img)

# ss = selectivesearch.selective_search(img)
init_segments = segmentation.felzenszwalb(img, scale=8, sigma=0.8, min_size=1000)
# plt.imshow(init_segments, cmap='Dark2')

# plt.figure()
# plt.imshow(img)
# rect = Rectangle((100, 100), 60, 80, fill=False, color='red', linewidth=2.0)
# plt.axes().add_patch(rect)
# plt.show()

# %% extract regions

def extract_regions(init_segments, img):
    R = {}

    for r in range(init_segments.max()+1):
