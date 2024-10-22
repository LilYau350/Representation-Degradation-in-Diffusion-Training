from PIL import Image
from argparse import ArgumentParser
import os
from multiprocessing import Pool
from tqdm import tqdm  
import numpy as np
import math
import random

# For older versions of Pillow, define Resampling as an alias for Image
try:
    from PIL import Resampling
except ImportError:
    class Resampling:
        LANCZOS = Image.LANCZOS
        NEAREST = Image.NEAREST
        BILINEAR = Image.BILINEAR
        BICUBIC = Image.BICUBIC
        HAMMING = Image.HAMMING
        BOX = Image.BOX

alg_dict = {
    'lanczos': Resampling.LANCZOS,
    'nearest': Resampling.NEAREST,
    'bilinear': Resampling.BILINEAR,
    'bicubic': Resampling.BICUBIC,
    'hamming': Resampling.HAMMING,
    'box': Resampling.BOX
}

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images", required=True)
    parser.add_argument('-o', '--out_dir', help="Output directory for resized images", required=True)
    parser.add_argument('-s', '--size', help="Size of an output image (e.g. 32 results in (32x32) image)",
                        default=64, type=int)
    parser.add_argument('-a', '--algorithm', help="Algorithm used for resampling: lanczos, nearest,"
                                                  " bilinear, bicubic, box, hamming",
                        default='box')
    parser.add_argument('-r', '--recurrent', help="Process all subfolders in this folder (1 lvl deep)",
                        action='store_true', default=0)
    parser.add_argument('-f', '--full', help="Use all algorithms, create subdirectory for each algorithm output",
                        action='store_true')
    parser.add_argument('-e', '--every_nth', help="Use if you don't want to take all classes, "
                                                  "if -e 10 then takes every 10th class",
                        default=1, type=int)
    parser.add_argument('-j', '--processes', help="Number of sub-processes that run different folders "
                                                  "in the same time ",
                        default=4, type=int)
    args = parser.parse_args()

    return args.in_dir, args.out_dir, args.algorithm, args.size, args.recurrent, \
           args.full, args.every_nth, args.processes

def str2alg(str):
    str = str.lower()
    return alg_dict.get(str, None)

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the reducing_gap
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def resize_img_folder(in_dir, out_dir, alg, size):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_list = [f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for filename in file_list:
        try:
            im = Image.open(os.path.join(in_dir, filename))

            # Convert grayscale images into 3 channels
            if im.mode != "RGB":
                im = im.convert(mode="RGB")

            # Apply center cropping to maintain aspect ratio without distortion
            cropped_arr = center_crop_arr(im, size)
            im_resized = Image.fromarray(cropped_arr)
            filename = os.path.splitext(filename)[0]
            output_path = os.path.join(out_dir, filename + '.png')
            im_resized.save(output_path)
        except OSError as err:
            print("This file couldn't be read as an image")
            with open("log.txt", "a") as f:
                f.write("Couldn't resize: %s\n" % os.path.join(in_dir, filename))

if __name__ == '__main__':
    in_dir, out_dir, alg_str, size, recurrent, full, every_nth, processes = parse_arguments()

    print('Starting ...')

    if full is False:
        algs = [alg_str]
    else:
        algs = alg_dict.keys()

    pool = Pool(processes=processes)

    repeat = False
    if recurrent:
        folders = [dir for dir in sorted(os.listdir(in_dir)) if os.path.isdir(os.path.join(in_dir, dir))]
        with tqdm(total=len(folders)) as pbar:  
            for i, folder in enumerate(folders):
                if i % every_nth == 0 or repeat is True:
                    resize_img_folder(os.path.join(in_dir, folder), os.path.join(out_dir, folder), alg_str, size)
                pbar.update(1)  
    else:
        resize_img_folder(in_dir=in_dir, out_dir=out_dir, alg=alg_str, size=size)

    pool.close()
    pool.join()
    print("Finished.")
