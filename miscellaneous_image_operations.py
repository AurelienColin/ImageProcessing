import numpy as np
import imutils
import fire
import os
import cv2
import tqdm
import glob

from Rignak_Misc.path import create_path

#python miscellaneous_image_operations.py --background=255 --input_folder=input/* --output_folder=output_misc --shape=(512,512)

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


def square_image(im, background=0):
    dim = np.max(im.shape[:2])
    if len(im.shape) == 3:
        square_im = np.zeros((dim, dim, 3)) + background
    else:
        square_im = np.zeros((dim, dim)) + background

    offset_x = (dim - im.shape[0]) // 2
    offset_y = (dim - im.shape[1]) // 2
    if offset_x != 0:
        square_im[offset_x:offset_x + im.shape[0]] = im
    elif offset_y != 0:
        square_im[:, offset_y:offset_y + im.shape[1]] = im
    else:
        square_im = im
    return square_im


def fourier_transform(image, remove_center=False):
    ftimage = np.fft.fft2(image)
    ftimage = np.fft.fftshift(ftimage)

    if remove_center:
        ftimage[image.shape[0] // 2] = 0
        ftimage[:, image.shape[1] // 2] = 0

    return ftimage, np.abs(np.real(ftimage)), np.abs(np.imag(ftimage))


def inverse_fourier_transform(ftimage, m=0):
    if ftimage.shape[-1] == 2:
        ftimage = ftimage[:, :, 0] + ftimage[:, :, 1] * 1j

    ftimage = np.fft.fftshift(ftimage)
    imagep = np.fft.ifft2(ftimage) + m
    return imagep.astype(np.float64)


def main(input_folder, output_folder, shape, background=0,
         authorized_extension=('.png', '.jpg', '.jpeg')):
    for path in tqdm.tqdm(glob.glob(input_folder)):
        split_path = os.path.splitext(path)
        if not split_path[1] or split_path[1] not in authorized_extension:
            continue
        end_path = os.path.join(output_folder, os.path.split(os.path.split(split_path[0])[0])[1], os.path.split(split_path[0])[-1] + '.png')
        create_path(end_path)

        im = cv2.imread(path)
        if im is None:
            print(f"{path} is invalid")
            continue
        im = square_image(im, background=background)

        im = imutils.resize(im, width=shape[0], height=shape[1], inter=cv2.INTER_CUBIC)
        cv2.imwrite(end_path, im)


if __name__ == '__main__':
    fire.Fire(main)
