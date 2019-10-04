import cv2
from lbpcascade_animeface.detect_face import detect as detect_anime_face
import os
from os.path import split, splitext, join
import sys
from tqdm import tqdm
import fire

from Rignak_ImageProcessing.miscellaneous_image_operations import extract_checked_bound
from Rignak_Misc.path import get_local_file

DETECTION_FUNCTION = detect_anime_face
PORTRAIT_MARGIN_FACTOR = 0.25
UPPER_BODY_MARGIN_FACTOR = 0.5

SUPPORTED_EXTENSION = ('.png', '.jpg')
OUTPUT_FOLDER = get_local_file(__file__, 'output')
INPUT_FOLDER = get_local_file(__file__, 'input')

FACE_MODE_FUNCTION = {
    "face": lambda im, x, y, width, height:
    extract_checked_bound(im,
                          int(x),
                          int(x + width),
                          int(y),
                          int(y + height)),
    "portrait": lambda im, x, y, width, height:
    extract_checked_bound(im,
                          int(x - width * PORTRAIT_MARGIN_FACTOR),
                          int(x + width + width * PORTRAIT_MARGIN_FACTOR),
                          int(y - height * PORTRAIT_MARGIN_FACTOR),
                          int(y + height + height * PORTRAIT_MARGIN_FACTOR)),
    "upper_body": lambda im, x, y, width, height:
    extract_checked_bound(im,
                          int(x - width * UPPER_BODY_MARGIN_FACTOR),
                          int(x + width + width * UPPER_BODY_MARGIN_FACTOR),
                          int(y),
                          int(y + height + height * UPPER_BODY_MARGIN_FACTOR * 2))
}

DEFAULT_MODES = FACE_MODE_FUNCTION.keys()


def extract_faces(full_filename, output_folder=OUTPUT_FOLDER, modes=DEFAULT_MODES):
    os.makedirs(output_folder, exist_ok=True)
    filename = split(full_filename)[-1]
    try:
        im = cv2.imread(full_filename, cv2.IMREAD_COLOR)
        for i, (x, y, width, height) in enumerate(DETECTION_FUNCTION(im)):
            for mode in modes:
                new_im = FACE_MODE_FUNCTION[mode](im, x, y, width, height)
                new_filename = join(output_folder, f"{splitext(filename)[0]}_{mode}.png")
                cv2.imwrite(new_filename, new_im)
    except Exception as e:
        print(f'Error on {filename}: {e}')


def main(*modes, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    """
    Detect faces and copy them to another folder

    :param input_folder: folder containing the images
    :param output_folder: future folder containing the faces
    :param modes: "face", "portrait", "upper_body"
    :return:
    """
    for filename in tqdm(os.listdir(input_folder)):
        if not os.path.splitext(filename)[-1] in SUPPORTED_EXTENSION:
            continue
        full_filename = join(input_folder, filename)
        extract_faces(full_filename, output_folder=output_folder, modes=modes)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(*DEFAULT_MODES)
    else:
        fire.Fire(main)
