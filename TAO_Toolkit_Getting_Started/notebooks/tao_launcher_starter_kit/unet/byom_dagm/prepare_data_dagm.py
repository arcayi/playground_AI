# Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.

"""Script to prepare train/val dataset for Unet tutorial."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2


def parse_args(args=None):
    """parse the arguments."""
    parser = argparse.ArgumentParser(description='Prepare train/val dataset for UNet tutorial')

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory to DAGM dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to DAGM dataset"
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default="Class7",
        help="Target Class for DAGM dataset"
    )
    return parser.parse_args(args)


def main(args=None):
    """Main function for data preparation."""

    args = parse_args(args)

    top_dir = os.path.join(args.input_dir, args.target_class)
    out_dir = os.path.join(args.output_dir, args.target_class)

    splits = ["Test", "Train"]

    for split in splits:
        print(f"Running {split}")
        imgs_lists, masks_lists = [], []

        img_dir = os.path.join(top_dir, f"{split}")
        mask_dir = os.path.join(top_dir, f"{split}/Label/")
        mask_file = os.path.join(mask_dir, "Labels.txt")

        with open(mask_file) as f:
            coordinates = f.read().split('\n')

        mask_shape = (512, 512, 3)
        for coord in tqdm(coordinates, total=len(coordinates)):
            if len(coord.split('\t')) >= 5:
                coord_split = coord.split('\t')
                filename = coord_split[0]
                has_label = int(coord_split[1])

                mask_img_file = os.path.join(mask_dir, f"{filename}_label.PNG")
                img_file = os.path.join(img_dir, f"{filename}.PNG")

                if not has_label:
                    blank_img = np.zeros(mask_shape)
                    cv2.imwrite(mask_img_file, blank_img)

                img_file = img_file.replace(top_dir, out_dir)
                mask_img_file = mask_img_file.replace(top_dir, out_dir)

                imgs_lists.append(img_file)
                masks_lists.append(mask_img_file)

        np.savetxt(os.path.join(top_dir, f"{split}_imgsall.txt".lower()), imgs_lists, fmt="%s")
        np.savetxt(os.path.join(top_dir, f"{split}_masksall.txt".lower()), masks_lists, fmt="%s")

    print("Prepared data successfully !")

if __name__ == "__main__":
    main()
