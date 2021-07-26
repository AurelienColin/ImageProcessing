import numpy as np
import os
import cv2
import imutils
from PIL import Image
import tqdm

BASE_FOLDER = "E:\\datasets\\outside_border\\train\\outside_border"

FIRST_FORMAT = {'folder': 'google', 'width': int(549 * 1.1), 'height': int(706 * 1.1), 'switch': 1}
SECOND_FORMAT = {'folder': 'wikipedia', 'width': 174, 'height': 839, 'switch': 0.3}
FORMATS = [FIRST_FORMAT, SECOND_FORMAT]


def extract_biggest_connected(im, v=255, crop=True, c=4):
    if len(im.shape) == 3 and im.shape[-1]:
        mask = abs(v - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    else:
        mask = abs(v - im)
    mask[mask > 1] = 255
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, 255, connectivity=c)

    im = im[:, :, [0, 1, 2, 0]]
    im[:, :, 3] = 255

    ims = [im.copy() for (_, _, _, _, area) in stats[1:] if area > 200 * 200]
    k = 0
    for i in range(nb_components):
        if i and stats[i][cv2.CC_STAT_AREA] > 200 * 200:
            ims[k][output != i] = 0
            ims[k] = ims[k][stats[i, 1]:stats[i, 1] + stats[i, 3],
                     stats[i, 0]:stats[i, 0] + stats[i, 2]]
            k += 1
    return ims
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    im[output != max_label] = 0
    if crop:
        im = im[stats[max_label, 1]:stats[max_label, 1] + stats[max_label, 3],
             stats[max_label, 0]:stats[max_label, 0] + stats[max_label, 2]]
    return [im]


def symmetry_on_border(im):
    mask = np.array(im)
    mask[mask > 1] = 255
    if np.sum(mask[:, 0]) > np.sum(mask[:, -1]):
        im = np.flip(im, axis=1)
    return im


def remove_transparency(namefile, background_color=(255, 255, 255)):
    im = Image.open(namefile)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, background_color)
        bg.paste(im, mask=alpha)
        bg.save(namefile)


def process_file(full_filename):
    remove_transparency(full_filename)
    # return
    folder, filename = os.path.split(full_filename)
    im = cv2.imread(full_filename)
    im_copy = im.copy()

    for i, subim in enumerate(extract_biggest_connected(im)):
        # subim = symmetry_on_border(subim)
        cv2.imwrite(f'{folder}/done/{os.path.splitext(filename)[0]}_{i}.png', subim)
        # os.rename(full_filename, os.path.join(folder, 'done', filename))  # backup

        ratio = im.shape[1] / im.shape[0]

        for format in FORMATS:
            if False and ratio <= format['switch']:
                im = imutils.resize(im, width=format['width'],
                                    height=format['height'], inter=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(folder, format['folder'],
                                         f"{os.path.splitext(filename)[0]}.jpg"), im)
                break


def process_folder(folder):
    for filename in tqdm.tqdm(os.listdir(folder)):
        try:
            process_file(os.path.join(folder, filename))
        except OSError:
            continue

if __name__ == '__main__':
    process_folder(BASE_FOLDER)
