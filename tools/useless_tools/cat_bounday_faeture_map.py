import os
import os.path as osp
import cv2
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)

    args = parser.parse_args()

    return args


def main():

    args = get_args()
    sub_file_names = os.listdir(args.file1)
    os.makedirs(args.save_file, exist_ok=True)
    for sub_file in sub_file_names:
        if sub_file == '.DS_Store':
            continue
        images = os.listdir(osp.join(args.file1, sub_file))
        boundaries = [token for token in images if 'boundary_' in token]
        masks = [token.replace('boundary_', 'mask_') for token in boundaries]
        num_images = len(masks)
        i = 0
        for boundary, mask in zip(boundaries, masks):
            i += 1
            save_image_name = osp.join(args.save_file, sub_file + '_' + mask)
            boundary1 = cv2.imread(osp.join(args.file1, sub_file, boundary))
            mask1 = cv2.imread(osp.join(args.file1, sub_file, mask))
            mask1 = cv2.resize(mask1, dsize=(28, 28))
            img1 = np.vstack([boundary1, mask1])
            boundary2 = cv2.imread(osp.join(args.file2, sub_file, boundary))
            mask2 = cv2.imread(osp.join(args.file2, sub_file, mask))
            mask2 = cv2.resize(mask2, dsize=(28, 28))
            img2 = np.vstack([boundary2, mask2])
            img = cv2.hconcat((img1, img2))
            cv2.imwrite(save_image_name, img)
            print(f'{mask} done, [{i}]/[{num_images}]')
        print(f'{sub_file} done')

if __name__ == '__main__':
    main()