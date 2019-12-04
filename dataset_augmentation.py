import numpy as np
import os
from PIL import Image

from Rignak_Misc.path import get_local_file
from Rignak_Misc.print import carriage_returned_print

ARGS = [(0,),  # vertical symmetry
        (-30, -20, -10, 0, 10, 20, 30),  # rotation
        (1,)]  # scale

INPUT_FOLDER = get_local_file(__file__, 'input')
OUTPUT_FOLDER = get_local_file(__file__, 'output')


def rotate(im, arg):
    return im.rotate(arg, Image.BICUBIC)


def symmetry(im, arg):
    if arg:
        im = Image.fromarray(np.array(im)[:, ::-1])
    return im


def scale(im, arg):
    w, h, c = np.array(im).shape
    return im.resize((int(h * arg), int(w * arg)), Image.BICUBIC)


def main():
    files = os.listdir(INPUT_FOLDER)
    n = len(files)
    for k, file in enumerate(files):
        for i, (sym, angle, factor) in enumerate([(sym, angle, factor)
                                                  for sym in ARGS[0]
                                                  for angle in ARGS[1]
                                                  for factor in ARGS[2]]):
            im = Image.open(os.path.join(INPUT_FOLDER, file))
            im = scale(symmetry(rotate(im, angle), sym), factor)
            new_file = f'{os.path.splitext(file)[0]}-{i}.png'
            im.save(os.path.join(OUTPUT_FOLDER, new_file))

        carriage_returned_print(f'{k}/{n}')


if __name__ == '__main__':
    main()
