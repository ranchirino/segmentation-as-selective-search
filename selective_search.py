import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import segmentation
from skimage import measure
from skimage.data import coffee
from datetime import datetime
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
    R = []
    for r in np.unique(init_segments):
        i = np.asarray(np.where(init_segments == r))
        x_min = i[1,:].min()
        x_max = i[1,:].max()
        y_min = i[0,:].min()
        y_max = i[0,:].max()
        width = (x_max - x_min) + 1
        height = (y_max - y_min) + 1
        R.append({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "width": width, "height": height, "size": i.shape[1], "label": r})
    return R

R = extract_regions(init_segments)

# plt.figure()
# plt.imshow(init_segments)
# for r in R.items():
#     d = r[1]
#     rect = Rectangle((d['x_min'], d['y_min']), d['width'], d['height'], fill=False, color='red', linewidth=1.0)
#     plt.axes().add_patch(rect)
# plt.show()

# plt.imshow(init_segments)
# plt.show()

# %% extract neighbors

def find_neighbours(reg_bb, label):
    reg = extract_regions(reg_bb)
    neig = []
    for r in reg:
        if r["label"] != label:
            window = reg_bb[r["y_min"]:r["y_max"]+1, r["x_min"]:r["x_max"]+1]
            if label in window:
                neig.append(r["label"])
    return neig

def extract_neighbors(seg_img, regions, img):
    N = []  # region, neighbours
    h = img.shape[0]  # rows
    w = img.shape[1]  # columns
    for r in regions:
        x_min = r["x_min"] - 1 if r["x_min"] != 0 else r["x_min"] # +1 padding
        x_max = r["x_max"] + 2 if r["x_max"] != w else r["x_max"] # +1 padding
        y_min = r["y_min"] - 1 if r["y_min"] != 0 else r["y_min"] # +1 padding
        y_max = r["y_max"] + 2 if r["y_max"] != h else r["y_max"] # +1 padding
        reg_bb = seg_img[y_min:y_max, x_min:x_max]
        neig = find_neighbours(reg_bb, r["label"])
        N.append({"region": r["label"], "neig": neig})
    return N

# t1 = datetime.now()
N = extract_neighbors(init_segments, R, img)
# t2 = datetime.now()
# print(t2-t1)

# bb_14 = init_segments[R[14]["y_min"]-1:R[14]["y_max"]+2, R[14]["x_min"]-1:R[14]["x_max"]+2]
# zeros_arr[np.where(zeros_arr != 0)] = 1
# plt.imshow(zeros_arr)
# plt.show()
# np.unique(zeros_arr[np.where(zeros_arr != 0)])

# contours = measure.find_contours(zeros_arr, 0)
# cont = np.round(np.concatenate(contours)).astype('int32')
# plt.imshow(contours)
# plt.show()

# zeros_arr[cont[:,0], cont[:,1]] = 20
# props = measure.regionprops(zeros_arr)
# zeros_arr[props[0].coords[:,0], props[0].coords[:,1]] = 20
# plt.imshow(zeros_arr)
# plt.show()

# n = []
# for cont in contours:
#     c = np.round(cont).astype('int32')
#     pix_cont = c[random.randint(len(c))]
#     window = zeros_arr[pix_cont[0]-1:pix_cont[0]+2, pix_cont[1]-1:pix_cont[1]+2]
#     reg = np.unique(window)
#     if 0 in reg:
#         n.append(reg[(reg != 20) & (reg != 0)])


#%% calculate similarity

def calc_sim(r1, r2, img):
    img_size = img.shape[0] * img.shape[1]
    r1 = [x for x in R if x['label'] == r1]
    r2 = [x for x in R if x['label'] == r2]



# def sim_size():
    # calculate the size similarity over the image

[x for x in R if x['label'] == 13]