"""
Multi-Image DECA Face Reconstruction
Modified version that uses multiple images of the same person for improved reconstruction
"""
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
        
    def reconstruct_multi_image(self, image_paths, output_dir, strategy='average'):
        """
        Reconstruct face using multiple images
        
        Args:
            image_paths: List of paths to images of the same person
            output_dir: Directory to save results
            strategy: 'average', 'weighted', or 'best'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        all_codes = []
        all_details = []
        all_images = []
        
        print(f"Processing {len(image_paths)} images...")
        for idx, img_path in enumerate(image_paths):
            print(f"Processing image {idx+1}/{len(image_paths)}: {img_path}")
            
            # Load and preprocess image
            image = imread(img_path)
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=2)
            if image.shape[2] == 4:
                image = image[:,:,:3]
                
            # Prepare input
            image_tensor = torch.tensor(image).float()
            image_tensor = image_tensor.permute(2,0,1).unsqueeze(0)
            image_tensor = image_tensor / 255.0
            
            if self.config.use_gpu:
                image_tensor = image_tensor.cuda()
            
            # Encode image
            with torch.no_grad():
                codedict = self.deca.encode(image_tensor)
                
            all_codes.append(codedict)
            all_images.append(image_tensor)
            
        # Merge reconstructions based on strategy
        merged_code = self.merge_codes(all_codes, strategy)
        
        # Decode with merged parameters
        with torch.no_grad():
            opdict, visdict = self.deca.decode(merged_code)
        
        # Save results
        self.save_results(opdict, visdict, output_dir, all_images)
        
        return opdict, visdict
    
    def merge_codes(self, all_codes, strategy='average'):
        """
        Merge multiple code dictionaries with advanced strategies
        """
        merged = {}
        
        if strategy == 'average':
            # Intelligent averaging with outlier removal
            for key in all_codes[0].keys():
                stacked = torch.stack([code[key] for code in all_codes])
                
                # Remove outliers using median absolute deviation
                if len(all_codes) >= 3:
                    median = stacked.median(dim=0)[0]
                    mad = torch.abs(stacked - median).median(dim=0)[0]
                    # Keep values within 2.5 MAD
                    mask = torch.abs(stacked - median) <= 2.5 * mad
                    # Use mean of filtered values
                    merged[key] = (stacked * mask.float()).sum(dim=0) / mask.float().sum(dim=0).clamp(min=1)
                else:
                    merged[key] = stacked.mean(dim=0)
                
        elif strategy == 'weighted':
            # Weight by reconstruction quality metrics
            weights = self.compute_weights(all_codes)
            weights_tensor = torch.tensor(weights, device=all_codes[0]['shape'].device)
            weights_tensor = weights_tensor / weights_tensor.sum()
            
            for key in all_codes[0].keys():
                stacked = torch.stack([code[key] for code in all_codes])
                # Apply weights across batch dimension
                weighted = stacked * weights_tensor.view(-1, *([1] * (stacked.dim()-1)))
                merged[key] = weighted.sum(dim=0)
                
        elif strategy == 'best':
            # Select best image based on multiple quality metrics
            best_idx = self.select_best_image(all_codes)
            merged = all_codes[best_idx]
            
        # Consistency constraints for identity-related parameters
        # Shape (identity) should be averaged across all images
        if 'shape' in merged:
            shape_stack = torch.stack([code['shape'] for code in all_codes])
            merged['shape'] = shape_stack.mean(dim=0)
        
        # Detail code (person-specific) should also be consistent
        if 'detail' in merged:
            detail_stack = torch.stack([code['detail'] for code in all_codes])
            merged['detail'] = detail_stack.mean(dim=0)
            
        return merged
    
    def compute_weights(self, all_codes):
        """
        Compute confidence weights for each reconstruction
        Simple version: uniform weights (can be enhanced with quality metrics)
        """
        return [1.0 / len(all_codes)] * len(all_codes)
    
    def select_best_image(self, all_codes):
        """
        Select the best image based on reconstruction quality
        """
        # Simple heuristic: use first image (can be enhanced)
        return 0
    
    def save_results(self, opdict, visdict, output_dir, images):
        """
        Save reconstruction results
        """
        # Save mesh
        if 'verts' in opdict:
            util.write_obj(
                os.path.join(output_dir, 'multi_image_mesh.obj'),
                opdict['verts'][0].cpu().numpy(),
                self.deca.flame.faces
            )
        
        # Save visualization
        if 'shape_images' in visdict:
            util.save_image(
                visdict['shape_images'][0],
                os.path.join(output_dir, 'shape.png')
            )
            
        if 'detail_images' in visdict:
            util.save_image(
                visdict['detail_images'][0],
                os.path.join(output_dir, 'detail.png')
            )
        
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
    
    # Additional DECA parameters
    parser.add_argument('--saveDepth', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() == 'true')
    
    args = parser.parse_args()
    
    # Setup configuration
    from yacs.config import CfgNode as CN
    cfg = CN(new_allowed=True)
    cfg.use_gpu = args.device == 'cuda' and torch.cuda.is_available()
    
    # Load DECA config if exists
    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)
    
    # Get all images from input folder
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
    
    # Initialize and run multi-image reconstruction
    multi_deca = MultiImageDECA(cfg)
    opdict, visdict = multi_deca.reconstruct_multi_image(
        image_paths,
        args.output_folder,
        strategy=args.strategy
    )
    
    print("Multi-image reconstruction complete!")


if __name__ == '__main__':
    main()
