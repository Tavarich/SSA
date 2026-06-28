import os
import numpy as np
from PIL import Image


def merge_masks(mask_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    palette_img = os.path.join("datasets/ref-davis", "palette.png")
    palette = Image.open(palette_img).getpalette()

    # Loop through each video in the mask directory
    for video in os.listdir(mask_dir):
        video_path = os.path.join(mask_dir, video)
        if os.path.isdir(video_path):
            # Group exps in chunks of 4 and merge accordingly
            exps = sorted(os.listdir(video_path), key=lambda x: int(x))
            num_exps = len(exps)

            # TODO checkpoint
            mask_files = sorted(os.listdir(os.path.join(video_path, exps[0])))  # because same frame masks
            num_frames = len(mask_files)

            for j in range(4):
                merged_mask = [None] * num_frames

                for i in range(j, num_exps, 4):
                    exp_path = os.path.join(video_path, exps[i])

                    for idx, mask in enumerate(mask_files):  # num_frames
                        mask_path = os.path.join(exp_path, mask)
                        mask = np.array(Image.open(mask_path))

                        if merged_mask[idx] is None:
                            merged_mask[idx] = np.zeros_like(mask)

                        # merged_mask[idx][mask > 0] = (i // 4) + 1
                        label = (i // 4) + 1
                        empty = (merged_mask[idx] == 0)
                        merged_mask[idx][empty & (mask > 0)] = label

                output_video_dir = os.path.join(output_dir, f'anno_{j}', video)
                os.makedirs(output_video_dir, exist_ok=True)
                for idx, mask in enumerate(mask_files):
                    merged_mask_img = Image.fromarray(merged_mask[idx]).convert('P')
                    merged_mask_img.putpalette(palette)
                    merged_mask_img.save(os.path.join(output_video_dir, mask))

                print(f' {video} has merged in {os.path.join(output_video_dir, mask)}')


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mask_dir', type=str, help='path of masks from clip', default='')
parser.add_argument('--output_dir', type=str, help='path of processed masks', default='')

args = parser.parse_args()

mask_dir = args.mask_dir
output_dir = args.output_dir

merge_masks(mask_dir, output_dir)
