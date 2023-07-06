import argparse
import numpy as np
import cv2
import os
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_cat_file', required=True)
    parser.add_argument('--gt_file', required=True)
    parser.add_argument('--save_file', required=True)

    args = parser.parse_args()
    return args


def main():

    args = get_args()
    os.makedirs(args.save_file, exist_ok=True)
    all_cat_images = os.listdir(args.all_cat_file)
    gt_images = [token.replace('boundary_bone_', '') for token in all_cat_images]
    gt_images = [token.replace('_leftImg8bit.png', '_gtFine_color.png') for token in gt_images]

    num_images = len(all_cat_images)
    for idx, all_cat_image, gt_image in zip(range(num_images), all_cat_images, gt_images):
        save_path = osp.join(args.save_file, gt_image)
        img1 = cv2.imread(osp.join(args.all_cat_file, all_cat_image))
        img2 = cv2.imread(osp.join(args.gt_file, gt_image))
        img2 = cv2.resize(img2, (512, 256))
        save_images = np.vstack([img1, img2])
        cv2.imwrite(save_path, save_images)
        print(f'{gt_image} done, [{idx}/{num_images}]')


if __name__ == '__main__':
    main()