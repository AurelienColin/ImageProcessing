import cv2
from lbpcascade_animeface.detect_face import detect as detect_anime_face
import os
from os.path import split, splitext, join
import sys
from tqdm import tqdm

from Rignak_ImageProcessing.miscellaneous_image_operations import extract_checked_bound
from Rignak_Misc.path import get_local_file

"""
Use: 
>>> python face_detection.py {input_folder} {output_folder} {mode1} {optional_mode2} {optional_mode3}
"""

DETECTION_FUNCTION = detect_anime_face
PORTRAIT_MARGIN_FACTOR = 0.25
UPPER_BODY_MARGIN_FACTOR = 0.5

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

DEFAULT_MODES = ("face",)


def parse_inputs(argvs):
    input_folder = argvs[1]
    output_folder = argvs[2]
    modes = argvs[3:]
    return input_folder, output_folder, modes


def extract_faces(full_filename, output_folder=OUTPUT_FOLDER, modes=DEFAULT_MODES):
    filename = split(full_filename)[-1]
    im = cv2.imread(full_filename, cv2.IMREAD_COLOR)
    for i, (x, y, width, height) in enumerate(DETECTION_FUNCTION(im)):
        for mode in modes:
            new_im = FACE_MODE_FUNCTION[mode](im, x, y, width, height)
            new_filename = join(output_folder, f"{splitext(filename)[0]}_{mode}.png")
            cv2.imwrite(new_filename, new_im)


def main(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, modes=DEFAULT_MODES):
    for file in tqdm(os.listdir(input_folder)):
        full_filename = join(input_folder, file)
        extract_faces(full_filename, output_folder=output_folder, modes=modes)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        input_folder, output_folder, modes = parse_inputs(sys.argv)
        main(input_folder=input_folder, output_folder=output_folder, modes=modes)
    else:
        main()
