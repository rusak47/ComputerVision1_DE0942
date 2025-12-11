# Patched multi_image_deca.py
# Fixed for handling multiple viewpoints (frontal + profile)

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
        """Use DECA's native preprocessing pipeline"""
        testdata = datasets.TestData(img_path)
        image_tensor = testdata[0]['image'].unsqueeze(0)
        if self.config.use_gpu:
            image_tensor = image_tensor.cuda()
        return image_tensor

    def select_best_frontal_view(self, all_codes):
        """Select the most frontal view based on pose parameters"""
        best_idx = 0
        min_rotation = float('inf')

        for idx, code in enumerate(all_codes):
            # Pose parameters: [global_rot (3), jaw_pose (3)]
            pose = code['pose'][0]  # Get first batch element

            # Calculate rotation magnitude (exclude jaw pose)
            global_rot = pose[:3]
            rotation_magnitude = torch.norm(global_rot).item()

            print(f"Image {idx + 1}: rotation magnitude = {rotation_magnitude:.4f}")

            if rotation_magnitude < min_rotation:
                min_rotation = rotation_magnitude
                best_idx = idx

        print(f"\nSelected image {best_idx + 1} as most frontal view")
        return best_idx

    def reconstruct_multi_image(self, image_paths, output_dir, strategy='identity'):
        """Reconstruct 3D face from multiple images"""
        os.makedirs(output_dir, exist_ok=True)

        all_codes = []
        all_images = []

        print(f"Processing {len(image_paths)} images...")
        for idx, img_path in enumerate(image_paths):
            print(f"\nProcessing image {idx + 1}/{len(image_paths)}: {img_path}")

            # Preprocess using DECA face detection / cropping
            image_tensor = self.preprocess_image(img_path)

            with torch.no_grad():
                codedict = self.deca.encode(image_tensor)

            all_codes.append(codedict)
            all_images.append(image_tensor)

        # Merge codes from all images
        merged_code = self.merge_codes(all_codes, strategy)

        # Decode to get 3D reconstruction
        with torch.no_grad():
            opdict, visdict = self.deca.decode(merged_code)

        # Save results
        self.save_results(opdict, visdict, output_dir, all_images)

        # Also save individual reconstructions for comparison
        self.save_individual_results(all_codes, output_dir)

        return opdict, visdict

    def merge_codes(self, all_codes, strategy='identity'):
        """
        Merge parameter codes from multiple images

        Strategy options:
        - 'identity': Average only identity features (shape, detail), use best frontal for pose/expression
        - 'frontal': Use the most frontal view entirely
        - 'average': Average all parameters (may cause artifacts with different poses)
        """
        merged = {}

        # Find the most frontal view
        frontal_idx = self.select_best_frontal_view(all_codes)
        frontal_code = all_codes[frontal_idx]

        if strategy == 'identity':
            print("\nUsing 'identity' merging strategy:")
            print("- Averaging shape and detail across all views")
            print("- Using frontal view for pose, expression, and lighting")

            # Start with frontal view
            merged = {key: frontal_code[key].clone() for key in frontal_code.keys()}

            # Average shape (identity feature) across all views
            if 'shape' in merged:
                shapes = torch.stack([code['shape'] for code in all_codes])
                merged['shape'] = shapes.mean(dim=0)
                print(f"  - Averaged shape from {len(all_codes)} views")

            # Average detail (identity feature) across all views
            if 'detail' in merged:
                details = torch.stack([code['detail'] for code in all_codes])
                merged['detail'] = details.mean(dim=0)
                print(f"  - Averaged detail from {len(all_codes)} views")

            # Keep pose, expression, camera, light from frontal view
            for key in ['pose', 'exp', 'cam', 'light']:
                if key in frontal_code:
                    merged[key] = frontal_code[key]

        elif strategy == 'frontal':
            print("\nUsing 'frontal' merging strategy:")
            print("- Using only the most frontal view")
            merged = {key: frontal_code[key].clone() for key in frontal_code.keys()}

        elif strategy == 'average':
            print("\nUsing 'average' merging strategy:")
            print("- Averaging ALL parameters (may cause artifacts)")

            for key in all_codes[0].keys():
                if key == 'images':
                    merged[key] = frontal_code[key]
                    continue

                stacked = torch.stack([code[key] for code in all_codes])
                merged[key] = stacked.mean(dim=0)

        # Always use frontal image for rendering
        merged['images'] = frontal_code['images']

        return merged

    def save_individual_results(self, all_codes, output_dir):
        """Save individual reconstructions for comparison"""
        individual_dir = os.path.join(output_dir, 'individual_views')
        os.makedirs(individual_dir, exist_ok=True)

        print(f"\nSaving individual reconstructions to {individual_dir}")

        for idx, codedict in enumerate(all_codes):
            with torch.no_grad():
                opdict, visdict = self.deca.decode(codedict)

            # Save mesh
            if 'verts' in opdict:
                faces = self.deca.render.faces[0].cpu().numpy()
                obj_path = os.path.join(individual_dir, f'view_{idx + 1}.obj')
                util.write_obj(
                    obj_path,
                    opdict['verts'][0].cpu().numpy(),
                    faces
                )

            # Save visualization
            if 'shape_images' in visdict:
                shape_img = visdict['shape_images'][0].detach().cpu()
                shape_img = shape_img.permute(1, 2, 0).numpy()
                shape_img = np.clip(shape_img * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(individual_dir, f'view_{idx + 1}_shape.png'),
                    cv2.cvtColor(shape_img, cv2.COLOR_RGB2BGR)
                )

    def save_results(self, opdict, visdict, output_dir, images):
        """Save reconstruction results"""

        print("\nSaving merged reconstruction results...")

        # Save mesh using DECA's method if texture is enabled
        output_path = os.path.join(output_dir, 'merged_mesh.obj')

        if self.config.model.use_tex:
            try:
                self.deca.save_obj(output_path, opdict)
                print(f"✓ Saved textured mesh to {output_path}")
                print(f"✓ Saved detailed mesh to {output_path.replace('.obj', '_detail.obj')}")
            except Exception as e:
                print(f"Error saving textured obj: {e}")
                print("Falling back to geometry-only save...")
                if 'verts' in opdict:
                    faces = self.deca.render.faces[0].cpu().numpy()
                    util.write_obj(output_path, opdict['verts'][0].cpu().numpy(), faces)
                    print(f"✓ Saved geometry-only mesh to {output_path}")
        else:
            # No texture - just save geometry
            if 'verts' in opdict:
                faces = self.deca.render.faces[0].cpu().numpy()
                util.write_obj(output_path, opdict['verts'][0].cpu().numpy(), faces)
                print(f"✓ Saved geometry mesh to {output_path}")

        # Save visualizations
        viz_saved = []

        if 'shape_images' in visdict:
            shape_img = visdict['shape_images'][0].detach().cpu()
            shape_img = shape_img.permute(1, 2, 0).numpy()
            shape_img = np.clip(shape_img * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'merged_shape.png'),
                        cv2.cvtColor(shape_img, cv2.COLOR_RGB2BGR))
            viz_saved.append('shape')

        if 'shape_detail_images' in visdict:
            detail_img = visdict['shape_detail_images'][0].detach().cpu()
            detail_img = detail_img.permute(1, 2, 0).numpy()
            detail_img = np.clip(detail_img * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'merged_detail.png'),
                        cv2.cvtColor(detail_img, cv2.COLOR_RGB2BGR))
            viz_saved.append('detail')

        if 'rendered_images' in visdict:
            rendered_img = visdict['rendered_images'][0].detach().cpu()
            rendered_img = rendered_img.permute(1, 2, 0).numpy()
            rendered_img = np.clip(rendered_img * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'merged_rendered.png'),
                        cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
            viz_saved.append('rendered')

        if 'landmarks2d' in visdict:
            lmk_img = visdict['landmarks2d'][0].detach().cpu()
            lmk_img = lmk_img.permute(1, 2, 0).numpy()
            lmk_img = np.clip(lmk_img * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, 'merged_landmarks.png'),
                        cv2.cvtColor(lmk_img, cv2.COLOR_RGB2BGR))
            viz_saved.append('landmarks')

        if viz_saved:
            print(f"✓ Saved visualizations: {', '.join(viz_saved)}")

        print(f"\n{'=' * 60}")
        print(f"All results saved to: {output_dir}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Image DECA Reconstruction')

    parser.add_argument('-i', '--input_folder', required=True,
                        help='Folder containing multiple images of the same person')
    parser.add_argument('-o', '--output_folder', default='results/multi_image',
                        help='Output folder for results')
    parser.add_argument('--strategy', default='identity',
                        choices=['identity', 'frontal', 'average'],
                        help='Merging strategy:\n'
                             '  identity: Average shape/detail, use frontal for pose (RECOMMENDED)\n'
                             '  frontal: Use only the most frontal view\n'
                             '  average: Average all parameters (may cause artifacts)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--config', default='configs/deca_config.yml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    from yacs.config import CfgNode as CN
    cfg = CN(new_allowed=True)
    cfg.use_gpu = args.device == 'cuda' and torch.cuda.is_available()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)
    else:
        print(f"Config file not found: {args.config}")
        return

    # Check texture configuration
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'use_tex'):
        if cfg.model.use_tex:
            print("✓ Texture support enabled")
        else:
            print("ℹ Texture support disabled (geometry only)")

    # Find all images in input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(args.input_folder).glob(f'*{ext}'))
        image_paths.extend(Path(args.input_folder).glob(f'*{ext.upper()}'))

    image_paths = sorted([str(p) for p in image_paths])

    if len(image_paths) == 0:
        print(f"No images found in {args.input_folder}")
        return

    print(f"\n{'=' * 60}")
    print(f"Multi-Image DECA Reconstruction")
    print(f"{'=' * 60}")
    print(f"Found {len(image_paths)} images")
    print(f"Strategy: {args.strategy}")
    print(f"{'=' * 60}\n")

    # Run multi-image reconstruction
    multi_deca = MultiImageDECA(cfg)
    multi_deca.reconstruct_multi_image(image_paths, args.output_folder, strategy=args.strategy)

    print("\n✓ Multi-image reconstruction complete!")


if __name__ == '__main__':
    main()