import os
import os.path as osp
import cv2
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)

    args = parser.parse_args()

    return args


def main():

    args = get_args()
    image_names_ori = os.listdir(args.file1)
    image_names2_ori = os.listdir(args.file2)
    image_names1 = [token for token in image_names_ori if '_gtFine_color.png' in token]
    os.makedirs(args.save_file, exist_ok=True)
    num_images = len(image_names1)
    for i, image_name in enumerate(image_names1):
        if image_name == '.DS_Store':
            continue
        img_name2 = 'color_mask_' + image_name.rsplit('_', 2)[0] + '_leftImg8bit.png'
        save_image_name = osp.join(args.save_file, img_name2)
        img1 = cv2.imread(osp.join(args.file1, image_name))
        img2 = cv2.imread(osp.join(args.file2, img_name2))
        img = np.vstack((img1, img2))
        cv2.imwrite(save_image_name, img)
        print(f'[{i}/{num_images}] {image_name} done')

if __name__ == '__main__':
    main()