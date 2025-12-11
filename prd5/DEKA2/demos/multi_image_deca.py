# Patched multi_image_deca.py
# Fixed for handling multiple viewpoints (frontal + profile)
# Fixed for texture extraction from input images

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
        # CRITICAL: The merged_code['images'] must contain the frontal image
        # for proper texture extraction
        merged_code = self.merge_codes(all_codes, strategy)

        # Decode to get 3D reconstruction with full outputs for texture
        print("\nDecoding merged parameters...")
        with torch.no_grad():
            # Decode with all necessary flags for texture
            opdict, visdict = self.deca.decode(
                merged_code,
                rendering=True,
                vis_lmk=True,
                return_vis=True,  # This generates uv_texture_gt needed for texture
                use_detail=True
            )

        # Debug: Print what's in opdict
        print("\nAvailable data in opdict:")
        for key in opdict.keys():
            if isinstance(opdict[key], torch.Tensor):
                print(f"  - {key}: {opdict[key].shape}")
            else:
                print(f"  - {key}: {type(opdict[key])}")

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

            # Start with frontal view (includes frontal image for texture extraction)
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

            # Average texture parameters across all views for better texture quality
            if 'tex' in merged:
                textures = torch.stack([code['tex'] for code in all_codes])
                merged['tex'] = textures.mean(dim=0)
                print(f"  - Averaged texture from {len(all_codes)} views")

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

        # Always use frontal image for rendering and texture extraction
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
                # Check if we have the required texture data
                required_keys = ['verts', 'uv_texture_gt', 'uv_detail_normals', 'displacement_map', 'normals']
                missing_keys = [key for key in required_keys if key not in opdict]

                if missing_keys:
                    print(f"⚠ Missing texture data: {missing_keys}")
                    print("  Saving geometry-only mesh...")
                    if 'verts' in opdict:
                        faces = self.deca.render.faces[0].cpu().numpy()
                        util.write_obj(output_path, opdict['verts'][0].cpu().numpy(), faces)
                        print(f"✓ Saved geometry-only mesh to {output_path}")
                else:
                    print("✓ All texture data present, saving with DECA's save_obj method...")
                    self.deca.save_obj(output_path, opdict)
                    print(f"✓ Saved textured mesh to {output_path}")
                    print(f"✓ Saved detailed mesh to {output_path.replace('.obj', '_detail.obj')}")

                    # Also save the texture as a separate image file
                    if 'uv_texture_gt' in opdict:
                        texture = util.tensor2image(opdict['uv_texture_gt'][0])
                        cv2.imwrite(
                            os.path.join(output_dir, 'texture_map.png'),
                            cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
                        )
                        print(f"✓ Saved texture map to texture_map.png")

            except Exception as e:
                print(f"⚠ Error saving textured obj: {e}")
                import traceback
                traceback.print_exc()
                print("  Falling back to geometry-only save...")
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
        viz_mapping = {
            'inputs': 'input_reference.png',
            'shape_images': 'merged_shape.png',
            'shape_detail_images': 'merged_detail.png',
            'rendered_images': 'merged_rendered.png',
            'landmarks2d': 'merged_landmarks.png',
            'landmarks3d': 'merged_landmarks3d.png'
        }
        viz_saved = []

        for key, filename in viz_mapping.items():
            if key in visdict:
                img = visdict[key][0].detach().cpu()
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(output_dir, filename),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
            viz_saved.append(key)

        if viz_saved:
            print(f"✓ Saved visualizations: {', '.join(viz_saved)}")

        print(f"\n{'=' * 60}")
        print(f"All results saved to: {output_dir}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Image DECA Reconstruction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended for profile + frontal images
  python multi_image_deca2.py -i input_folder -o results --strategy identity

  # Use only frontal view
  python multi_image_deca2.py -i input_folder -o results --strategy frontal
        """
    )

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

    if not os.path.exists(args.config):
        print(f"✗ Config file not found: {args.config}")
        return

    cfg.merge_from_file(args.config)

    # Check texture configuration
    print("\n" + "=" * 60)
    print("Configuration Check")
    print("=" * 60)
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'use_tex'):
        if cfg.model.use_tex:
            print("✓ Texture support: ENABLED")
            if hasattr(cfg.model, 'tex_type'):
                print(f"  - Texture type: {cfg.model.tex_type}")
            if hasattr(cfg.model, 'extract_tex'):
                print(f"  - Extract texture: {cfg.model.extract_tex}")
        else:
            print("ℹ Texture support: DISABLED (geometry only)")
            print("  To enable: set use_tex: True in config")
    print("=" * 60 + "\n")

    # Find all images in input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(args.input_folder).glob(f'*{ext}'))
        image_paths.extend(Path(args.input_folder).glob(f'*{ext.upper()}'))

    image_paths = sorted([str(p) for p in image_paths])

    if len(image_paths) == 0:
        print(f"✗ No images found in {args.input_folder}")
        return

    print(f"\n{'=' * 60}")
    print(f"Multi-Image DECA Reconstruction")
    print(f"{'=' * 60}")
    print(f"Found {len(image_paths)} images")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output_folder}\n")

    # Run multi-image reconstruction
    multi_deca = MultiImageDECA(cfg)
    multi_deca.reconstruct_multi_image(image_paths, args.output_folder, strategy=args.strategy)

    print("\n✓ Multi-image reconstruction complete!")


if __name__ == '__main__':
    main()