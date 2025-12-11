# Patched multi_image_deca.py
# Includes DECA-compatible preprocessing fixes

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decalib.deca import DECA
from decalib.utils import util
from decalib.datasets import datasets
from skimage.io import imread

class MultiImageDECA:
    def __init__(self, config):
        self.deca = DECA(config=config)
        self.config = config

    def preprocess_image(self, img_path):
        # Use DECA's native preprocessing pipeline
        testdata = datasets.TestData(img_path)
        image_tensor = testdata[0]['image'].unsqueeze(0)
        if self.config.use_gpu:
            image_tensor = image_tensor.cuda()
        return image_tensor

    def reconstruct_multi_image(self, image_paths, output_dir, strategy='average'):
        os.makedirs(output_dir, exist_ok=True)

        all_codes = []
        all_images = []

        print(f"Processing {len(image_paths)} images...")
        for idx, img_path in enumerate(image_paths):
            print(f"Processing image {idx+1}/{len(image_paths)}: {img_path}")

            # Preprocess using DECA face detection / cropping
            image_tensor = self.preprocess_image(img_path)

            with torch.no_grad():
                codedict = self.deca.encode(image_tensor)

            all_codes.append(codedict)
            all_images.append(image_tensor)

        merged_code = self.merge_codes(all_codes, strategy)

        with torch.no_grad():
            opdict, visdict = self.deca.decode(merged_code)

        self.save_results(opdict, visdict, output_dir, all_images)
        return opdict, visdict

    def merge_codes(self, all_codes, strategy='average'):
        merged = {}

        if strategy == 'average':
            for key in all_codes[0].keys():
                if key == 'images':
                    continue
                stacked = torch.stack([code[key] for code in all_codes])
                merged[key] = stacked.mean(dim=0)

        elif strategy == 'best':
            merged = all_codes[0]

        # Identity consistency
        if 'shape' in merged:
            merged['shape'] = torch.stack([code['shape'] for code in all_codes]).mean(dim=0)
        if 'detail' in merged:
            merged['detail'] = torch.stack([code['detail'] for code in all_codes]).mean(dim=0)

        merged['images'] = all_codes[0]['images']
        return merged

    def save_results(self, opdict, visdict, output_dir, images):
        if 'verts' in opdict:
            util.write_obj(
                os.path.join(output_dir, 'multi_image_mesh.obj'),
                opdict['verts'][0].cpu().numpy(),
                self.deca.flame.faces
            )

        if 'shape_images' in visdict:
            util.save_image(visdict['shape_images'][0], os.path.join(output_dir, 'shape.png'))

        if 'detail_images' in visdict:
            util.save_image(visdict['detail_images'][0], os.path.join(output_dir, 'detail.png'))

        print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Image DECA Reconstruction')

    parser.add_argument('-i', '--input_folder', required=True,
                        help='Folder containing multiple images of the same person')
    parser.add_argument('-o', '--output_folder', default='results/multi_image',
                        help='Output folder for results')
    parser.add_argument('--strategy', default='average',
                        choices=['average', 'weighted', 'best'],
                        help='Merging strategy: average, weighted, or best')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--config', default='configs/deca_config.yml',
                        help='Path to config file')

    args = parser.parse_args()

    from yacs.config import CfgNode as CN
    cfg = CN(new_allowed=True)
    cfg.use_gpu = args.device == 'cuda' and torch.cuda.is_available()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(args.input_folder).glob(f'*{ext}'))
        image_paths.extend(Path(args.input_folder).glob(f'*{ext.upper()}'))

    image_paths = sorted([str(p) for p in image_paths])

    if len(image_paths) == 0:
        print(f"No images found in {args.input_folder}")
        return

    print(f"Found {len(image_paths)} images")

    multi_deca = MultiImageDECA(cfg)
    multi_deca.reconstruct_multi_image(image_paths, args.output_folder, strategy=args.strategy)

    print("Multi-image reconstruction complete!")


if __name__ == '__main__':
    main()
