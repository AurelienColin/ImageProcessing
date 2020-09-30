import os
from os.path import split, splitext, join
import sys
import cv2
from tqdm import tqdm
import fire
import numpy as np
from lbpcascade_animeface.detect_face import detect as detect_anime_face

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.models import load_model
from skimage.transform import resize

from Rignak_ImageProcessing.miscellaneous_image_operations import extract_checked_bound, square_image
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


def get_is_face():
    model = load_model("nagadomi.h5", compile=False)
    input_shape = model.layers[0].input_shape[-3:]

    def is_face(im):
        im = resize(im, input_shape[:2])
        if input_shape[-1] == 1 and len(im.shape) == 3 and im.shape[-1] == 3:
            im = np.mean(im, axis=-1)
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=-1)
        categorization = np.argmax(model.predict(im)[0])
        return categorization == 1

    return is_face


is_face = get_is_face()


def extract_faces(full_filename, output_folder=OUTPUT_FOLDER, modes=DEFAULT_MODES, heatmap=False):
    os.makedirs(output_folder, exist_ok=True)
    if heatmap:
        assert len(modes) == 1, "Can't have multimodes for heatmap"
        os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'sources'), exist_ok=True)
    filename = split(full_filename)[-1]
    new_filenames = []
    try:
        im = cv2.imread(full_filename, cv2.IMREAD_COLOR)

        mask = np.zeros(im.shape[:2])

        detection = DETECTION_FUNCTION(im)
        for i, (x, y, width, height) in enumerate(detection):
            face = FACE_MODE_FUNCTION["face"](im, x, y, width, height)[0]
            if not is_face(face):
                continue
            
            for mode in modes:
                new_im, bounds = FACE_MODE_FUNCTION[mode](im, x, y, width, height)
                if heatmap:
                    mask[bounds[1]:bounds[3], bounds[0]:bounds[2]] = 255
                else:
                    new_im = square_image(new_im)
                    new_filename = join(output_folder, f"{splitext(filename)[0]}_{i}_{mode}.png")
                    cv2.imwrite(new_filename, new_im)
                    new_filenames.append(new_filename)
        if heatmap and np.mean(mask):
            new_filename = join(output_folder, 'masks', f"{splitext(filename)[0]}.png")
            cv2.imwrite(new_filename.replace('.png', '_heatmap.png'), mask)
            new_filename = join(output_folder, 'sources', f"{splitext(filename)[0]}.png")
            cv2.imwrite(new_filename.replace('.png', '_source.png'), im)
    except Exception as e:
        print(f'Error on {filename}: {e}')
        return []
    return new_filenames


def main(*modes, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, heatmap=False):
    """
    Detect faces and copy them to another folder

    :param input_folder: folder containing the images
    :param output_folder: future folder containing the faces
    :param modes: "face", "portrait", "upper_body"
    :return:
    """
    for filename in tqdm(os.listdir(input_folder)):
        # print(input_folder, filename)
        if not os.path.splitext(filename)[-1] in SUPPORTED_EXTENSION:
            continue
        full_filename = join(input_folder, filename)
        extract_faces(full_filename, output_folder=output_folder, modes=modes, heatmap=heatmap)


if __name__ == '__main__':
    fire.Fire(main)
