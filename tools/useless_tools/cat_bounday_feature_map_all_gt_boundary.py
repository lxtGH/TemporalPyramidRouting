import argparse
import numpy as np
import cv2
import os
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', required=True)
    parser.add_argument('--file2', required=True)
    parser.add_argument('--save_file', required=True)

    args = parser.parse_args()
    return args


def main():

    args = get_args()
    os.makedirs(args.save_file, exist_ok=True)
    pred_boundaries = os.listdir(args.file1)
    all_cat_images = [token.replace('pred_boundary', '') for token in pred_boundaries]
    all_cat_images = [token.replace('_leftImg8bit.png', '_gtFine_color.png') for token in all_cat_images]

    num_images = len(pred_boundaries)
    for idx, pred_boundary, all_cat_image in zip(range(num_images), pred_boundaries, all_cat_images):
        save_path = osp.join(args.save_file, pred_boundary)
        img1 = cv2.imread(osp.join(args.file1, pred_boundary))
        img1 = cv2.resize(img1, (512, 256))
        img2 = cv2.imread(osp.join(args.file2, all_cat_image))
        save_images = np.vstack([img1, img2])
        cv2.imwrite(save_path, save_images)
        print(f'{pred_boundary} done, [{idx}/{num_images}]')


if __name__ == '__main__':
    main()