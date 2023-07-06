import argparse
import numpy as np
import cv2
import os
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--save_file', required=True)

    args = parser.parse_args()
    return args


def main():

    args = get_args()
    os.makedirs(args.save_file, exist_ok=True)
    image_list = os.listdir(args.input_file)
    boundary_features = [token for token in image_list if 'boundary_bone' in token]
    # contraction_features = [token.replace('boundary_bone_', 'contraction_feature_bone') for token in boundary_features]
    # expansion_features = [token.replace('boundary_bone_', 'expansion_feature_bone') for token in boundary_features]
    contraction_warps = [token.replace('boundary_bone', 'contraction_warp') for token in boundary_features]
    expansion_warps = [token.replace('boundary_bone', 'expansion_warp') for token in boundary_features]
    overlaps = [token.replace('boundary_bone', 'overlap') for token in boundary_features]
    # color_maps = [token.replace('boundary_bone', 'color_mask') for token in boundary_features]

    num_images = len(boundary_features)
    for idx, boundary_feature_name, contraction_warp, expansion_warp, color_map \
        in zip(range(num_images), boundary_features, contraction_warps, expansion_warps, overlaps):
        save_path = osp.join(args.save_file, boundary_feature_name)
        boundary_feature = cv2.imread(osp.join(args.input_file, boundary_feature_name))
        # contraction_feature = cv2.imread(osp.join(args.input_file, contraction_feature))
        # expansion_feature = cv2.imread(osp.join(args.input_file, expansion_feature))
        contraction_warp = cv2.imread(osp.join(args.input_file, contraction_warp))
        expansion_warp = cv2.imread(osp.join(args.input_file, expansion_warp))
        color_map = cv2.imread(osp.join(args.input_file, color_map))
        color_map = cv2.resize(color_map, (320, 180))
        save_images = np.vstack([boundary_feature, contraction_warp, expansion_warp, color_map])
        cv2.imwrite(save_path, save_images)
        print(f'{boundary_feature_name} done, [{idx}/{num_images}]')


if __name__ == '__main__':
    main()