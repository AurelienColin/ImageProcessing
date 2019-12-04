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

def square_image(im):
    dim = np.max(im.shape[:2])
    if len(im.shape) == 3:
        square_im = np.zeros((dim, dim,3))
    else:
        square_im = np.zeros((dim, dim))

    offset_x = (dim-im.shape[0])//2
    offset_y = (dim-im.shape[1])//2
    if offset_x != 0:
        square_im[offset_x:offset_x+im.shape[0]] = im
    elif offset_y != 0:
        square_im[:, offset_y:offset_y+im.shape[1]] = im
    else:
        square_im = im
    return square_im