import numpy as np

def extract_checked_bound(im, x_min, x_max, y_min, y_max):
    x_min, y_min = check_bound(im, x_min, y_min)
    x_max, y_max = check_bound(im, x_max, y_max)
    return im[y_min:y_max, x_min:x_max]


def check_bound(im, x, y):
    if x < 0:
        x = 0
    elif x > im.shape[1]:
        x = im.shape[1] - 1
    if y < 0:
        y = 0
    elif y > im.shape[0]:
        y = im.shape[0] - 1
    return x, y


def median_subsampling(block):
    return np.median(block, axis=[2, 3])
