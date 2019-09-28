from scipy import signal
from PIL import Image
import sys
import numpy as np
import io
import os
import fire
import matplotlib.pyplot as plt

from Rignak_Misc.path import get_local_file

INPUT_FOLDER = get_local_file(__file__, 'input')
HIGH_PASS_FILTER = np.array([[0, -3, 0], [-3, 12, -3], [0, -3, 0]])
MIN_POSSIBLE_QUALITY = 1
MAX_POSSIBLE_QUALITY = 100


def filter_array(array, array_filter):
    return signal.convolve2d(array, array_filter)[1:-1, 1:-1]


def filter_image(im, array_filter=HIGH_PASS_FILTER):
    array = np.array(im)
    grayscale_array = np.mean(array, axis=-1)
    filtered_array = filter_array(grayscale_array, array_filter)
    return filtered_array


def jpeg_noise_detection(filename):
    jpeg_noise = []

    im = Image.open(filename)
    filtered_array = filter_image(im)

    for quality in range(MIN_POSSIBLE_QUALITY, MAX_POSSIBLE_QUALITY):
        new_noised_file = io.BytesIO()
        im.save(new_noised_file, format="JPEG", quality=quality)

        new_noised_image = Image.open(new_noised_file)
        new_filtered_array = filter_image(new_noised_image)

        array_subtraction = np.absolute(filtered_array - new_filtered_array)
        jpeg_noise.append((1 - np.mean(array_subtraction, axis=(0, 1)), quality))

    return jpeg_noise


def main(input_folder):
    for filename in os.listdir(input_folder):
        jpeg_noise = jpeg_noise_detection(os.path.join(input_folder, filename))
        plt.plot([e[1] for e in jpeg_noise],
                 [e[0] for e in jpeg_noise],
                 label=os.path.split(filename)[-1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(INPUT_FOLDER)
    else:
        fire.Fire(main)
