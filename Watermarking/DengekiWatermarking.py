import numpy as np
from PIL import Image
import os
import fire

INPUT_FOLDER = 'input'
IMAGE_INPUT_FOLDER = os.path.join(INPUT_FOLDER, 'watermarked')
OUTPUT_FOLDER = 'output'

WATERMARK_ON_BLACK_FILENAME = os.path.join(INPUT_FOLDER, 'watermarks', 'dengeki_on_black.jpg')
WATERMARK_ON_WHITE_FILENAME = os.path.join(INPUT_FOLDER, 'watermarks', 'dengeki_on_white.jpg')


def pixel_watermark_cleaning(pixel, watermark_on_black, watermark_on_white, l=100):
    watermark_on_black = 255 - watermark_on_black
    watermark_on_white = 255 - watermark_on_white

    cleaned_pixel = pixel.copy()
    for i in range(l):
        cleaned_pixel = pixel + (watermark_on_white * cleaned_pixel / 255) \
                        - (watermark_on_black * (255 - cleaned_pixel) / 255)
    return cleaned_pixel.astype(int)


def watermark_cleaning(filename,
                       watermark_on_white_filename=WATERMARK_ON_WHITE_FILENAME,
                       watermark_on_black_filename=WATERMARK_ON_BLACK_FILENAME,
                       output_folder=OUTPUT_FOLDER):
    im = Image.open(filename)
    watermark_on_white = Image.open(watermark_on_white_filename)
    watermark_on_black = Image.open(watermark_on_black_filename)

    width, height = im.size
    watermark_width, watermark_height = watermark_on_white.size

    cleaned_im = Image.new("RGB", (width, height))
    for y in range(height):
        cleaned_im.putpixel((0, y), im.getpixel((0, y)))
        for x in range(1, width):
            im_pixel = im.getpixel((x, y))
            if height - y > watermark_height or height - y <= 10:  # No watermark
                cleaned_pixel = im_pixel
            else:
                watermark_x = (x - 1) % watermark_width
                watermark_y = y - height + watermark_height
                cleaned_pixel = pixel_watermark_cleaning(np.array(im_pixel),
                                                         np.array(watermark_on_black.getpixel((watermark_x, watermark_y))),
                                                         np.array(watermark_on_white.getpixel((watermark_x, watermark_y))))
            cleaned_im.putpixel((x, y), tuple(cleaned_pixel))

    cleaned_im.save(os.path.join(output_folder, os.path.split(filename)[-1]))


def main(input_folder=IMAGE_INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    for filename in os.listdir(input_folder):
        watermark_cleaning(os.path.join(input_folder, filename), output_folder=output_folder)


if __name__ == '__main__':
    fire.Fire(main())
