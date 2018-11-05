import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import segmentation
from skimage import measure
from skimage.data import coffee
# import selectivesearch

img = coffee()
# plt.imshow(img)

# ss = selectivesearch.selective_search(img)
init_segments = segmentation.felzenszwalb(img, scale=10, sigma=0.8, min_size=1000)
# plt.imshow(init_segments, cmap='Dark2')

# plt.figure()
# plt.imshow(img)
# rect = Rectangle((100, 100), 60, 80, fill=False, color='red', linewidth=2.0)
# plt.axes().add_patch(rect)
# plt.show()

# %% extract regions

def extract_regions(init_segments):
    R = {}  # x_min, y_min, width, height, label

    for r in range(init_segments.max()+1):
        i = np.asarray(np.where(init_segments == r))
        x_min = i[1,:].min()
        x_max = i[1,:].max()
        y_min = i[0,:].min()
        y_max = i[0,:].max()
        width = (x_max - x_min) + 1
        height = (y_max - y_min) + 1
        R[r] = {"x_min": x_min, "y_min": y_min, "width": width, "height": height, "label": r}

    return R

R = extract_regions(init_segments)

plt.figure()
plt.imshow(init_segments)
for r in R.items():
    d = r[1]
    rect = Rectangle((d['x_min'], d['y_min']), d['width'], d['height'], fill=False, color='red', linewidth=1.0)
    plt.axes().add_patch(rect)
plt.show()

# plt.imshow(init_segments)
# plt.show()

# %% extract neighbors

# def extract_neighbors(regions):

zeros_arr = init_segments[R[0]["x_min"]:R[0]["height"]+1, R[0]["y_min"]:R[0]["width"]+1]
# zeros_arr[np.where(zeros_arr != 0)] = 1
plt.imshow(zeros_arr)
plt.show()
np.unique(zeros_arr[np.where(zeros_arr != 0)])

contours = measure.find_contours(zeros_arr, 0, fully_connected='high')
# cont = np.concatenate(contours)
cont = np.round(np.concatenate(contours)).astype('int32')
# plt.imshow(contours)
# plt.show()

# c1 = np.round(contours[0]).astype('int32')
# c2 = np.round(contours[1]).astype('int32')
# c3 = np.round(contours[2]).astype('int32')
# c4 = np.round(contours[3]).astype('int32')
# c5 = np.round(contours[4]).astype('int32')
zeros_arr[cont[:,0], cont[:,1]] = 20
# zeros_arr[c2[:,0], c2[:,1]] = 20
# zeros_arr[c3[:,0], c3[:,1]] = 20
# zeros_arr[c4[:,0], c4[:,1]] = 20
# zeros_arr[c5[:,0], c5[:,1]] = 20
# props = measure.regionprops(zeros_arr)
# zeros_arr[props[0].coords[:,0], props[0].coords[:,1]] = 20
plt.imshow(zeros_arr)
plt.show()

# n = []
# for cont in contours:
#     c = np.round(cont).astype('int32')
#     pix_cont = c[random.randint(len(c))]
#     window = zeros_arr[pix_cont[0]-1:pix_cont[0]+2, pix_cont[1]-1:pix_cont[1]+2]
#     reg = np.unique(window)
#     if 0 in reg:
#         n.append(reg[(reg != 20) & (reg != 0)])

# hacer un bounding box alrededor de cada region y ver si en ese bb se encuentra la region que estoy buscando, eso me dice si son vecinos o no
