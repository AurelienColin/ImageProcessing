import numpy as np
import os
import cv2
import imutils
from PIL import Image

BASE_FOLDER = "results"

FIRST_FORMAT = {'folder': 'google', 'width': int(549 * 1.1), 'height': int(706 * 1.1), 'switch': 1}
SECOND_FORMAT = {'folder': 'wikipedia', 'width': 174, 'height': 839, 'switch': 0.3}
FORMATS = [FIRST_FORMAT, SECOND_FORMAT]


def extract_biggest_connected(im):
    mask = 255 - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask[mask > 1] = 255
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, 255, connectivity=4)
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    im[output != max_label] = 255
    im = im[stats[max_label, 1]:stats[max_label, 1] + stats[max_label, 3],
            stats[max_label, 0]:stats[max_label, 0] + stats[max_label, 2]]
    return im


def symmetry_on_border(im):
    mask = np.array(im)
    mask[mask > 1] = 255
    if np.sum(mask[:, 0]) > np.sum(mask[:, -1]):
        im = np.flip(im, axis=1)
    return im


def remove_transparency(namefile, background_color=(128, 128, 128)):
    im = Image.open(namefile)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, background_color)
        bg.paste(im, mask=alpha)
        bg.save(namefile)


def process_file(full_filename):
    remove_transparency(full_filename)
    folder, filename = os.path.split(full_filename)[0]
    im = cv2.imread(full_filename)

    im = extract_biggest_connected(im)
    im = symmetry_on_border(im)
    os.rename(full_filename, os.path.join(folder, 'done', filename))  # backup

    ratio = im.shape[1] / im.shape[0]

    for format in FORMATS:
        if ratio <= format['switch']:
            im = imutils.resize(im, width=format['width'],
                                height=format['height'], inter=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(folder, format['folder'],
                                     f"{os.path.splitext(filename)[0]}.jpg"), im)
            break


def process_folder(folder):
    for filename in os.listdir(folder):
        process_file(os.path.join(folder, filename))


if __name__ == '__main__':
    process_folder(BASE_FOLDER)
