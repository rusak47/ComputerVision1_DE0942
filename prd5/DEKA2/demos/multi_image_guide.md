# Multi-Image DECA Face Reconstruction Guide

This guide explains how to modify DECA to use multiple images for improved face reconstruction.

## Overview

While DECA is designed to work with single images, using multiple images of the same person can significantly improve reconstruction quality by:

1. **Averaging identity parameters** across multiple views
2. **Reducing noise** in shape and detail estimation
3. **Capturing better detail consistency** 
4. **Handling occlusions** better through multi-view information

## Key Concepts

### DECA's Multi-Image Training
During training, DECA uses multiple images of the same person to learn detail consistency - separating person-specific details (moles, pores) from expression-dependent wrinkles. We can leverage similar principles for inference.

### Parameters to Merge

- **Shape (β)**: Identity-related, should be averaged across all images
- **Detail (δ)**: Person-specific details, should be consistent
- **Expression (ψ)**: Can vary per image, choose based on goal
- **Pose (θ)**: Can vary per image
- **Albedo (α)**: Texture, should be consistent for same person

## Implementation Strategies

### 1. Simple Averaging
Average all parameters across images. Best for images with similar expressions and poses.

```python
merged['shape'] = torch.stack([code['shape'] for code in all_codes]).mean(dim=0)
```

### 2. Robust Averaging with Outlier Removal
Use median absolute deviation to remove outliers before averaging.

```python
median = stacked.median(dim=0)[0]
mad = torch.abs(stacked - median).median(dim=0)[0]
mask = torch.abs(stacked - median) <= 2.5 * mad
merged[key] = (stacked * mask.float()).sum(dim=0) / mask.float().sum(dim=0)
```

### 3. Weighted Merging
Weight images based on quality metrics:
- Expression neutrality (for shape)
- Pose frontality
- Image quality
- Detection confidence

### 4. Selective Merging
- Average identity parameters (shape, detail, albedo)
- Keep expression and pose from target image
- Useful for animation with improved identity

## Usage

### Basic Usage

```bash
# Place multiple images of same person in a folder
python demo_multi_reconstruct.py -i path/to/images -o results/output

# Specify merging strategy
python demo_multi_reconstruct.py -i path/to/images -o results/output --strategy weighted

# Use best single image (with quality filtering)
python demo_multi_reconstruct.py -i path/to/images -o results/output --strategy best
```

### Advanced Usage

```python
from multi_image_deca import MultiImageDECA

# Initialize
multi_deca = MultiImageDECA(config)

# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
opdict, visdict = multi_deca.reconstruct_multi_image(
    image_paths,
    output_dir='results',
    strategy='weighted'
)

# Access results
vertices = opdict['verts']
faces = multi_deca.deca.flame.faces
```

## Best Practices

### Image Selection
1. **Quantity**: 3-10 images optimal (more doesn't always help)
2. **Diversity**: Mix of expressions and slight pose variations
3. **Quality**: Well-lit, in-focus images
4. **Consistency**: Same person, similar time period

### Processing Tips
1. **Identity Parameters**: Always average shape and detail codes
2. **Expression**: For neutral face, weight toward neutral expressions
3. **Pose**: For frontal face, weight toward frontal views
4. **Texture**: Average albedo for consistent appearance

### Quality Improvements
- Use images from different angles to handle occlusions
- Include at least one frontal image
- Avoid extreme expressions for shape estimation
- Filter out low-quality detections

## Alternative Approaches

### 1. Sequential Refinement
Instead of batch processing, refine reconstruction iteratively:

```python
def iterative_refinement(images):
    # Start with first image
    current_code = deca.encode(images[0])
    
    # Refine with subsequent images
    for img in images[1:]:
        new_code = deca.encode(img)
        # Blend identity parameters
        current_code['shape'] = 0.7 * current_code['shape'] + 0.3 * new_code['shape']
    
    return current_code
```

### 2. Bundle Adjustment Style
Optimize parameters to minimize reprojection error across all views:

```python
# Initialize with averaged parameters
merged_params = average_parameters(all_codes)

# Optimize to fit all images
optimizer = torch.optim.Adam([merged_params['shape']], lr=0.001)
for iteration in range(100):
    total_loss = 0
    for img, code in zip(images, all_codes):
        reconstructed = deca.decode({**code, 'shape': merged_params['shape']})
        loss = photometric_loss(img, reconstructed)
        total_loss += loss
    
    total_loss.backward()
    optimizer.step()
```

### 3. Consistency-Based Selection
Use DECA's detail consistency principle to select best subset:

```python
def select_consistent_images(all_codes):
    # Compute pairwise consistency
    consistency_matrix = compute_consistency(all_codes)
    
    # Select most consistent subset
    best_subset = greedy_selection(consistency_matrix)
    
    return [all_codes[i] for i in best_subset]
```

## Integration with Original Demo

To integrate into the original `demo_reconstruct.py`:

```python
# Original demo processes single image
for image_name in image_list:
    image = load_image(image_name)
    codedict = deca.encode(image)
    opdict, visdict = deca.decode(codedict)

# Modified for multi-image
image_group = group_images_by_person(image_list)
for person_id, person_images in image_group.items():
    # Collect codes
    codes = [deca.encode(load_image(img)) for img in person_images]
    
    # Merge codes
    merged_code = merge_codes(codes, strategy='average')
    
    # Decode merged
    opdict, visdict = deca.decode(merged_code)
```

## Limitations & Considerations

1. **Computational Cost**: Processing multiple images takes longer
2. **Alignment**: Images should be properly face-detected and aligned
3. **Same Person**: Algorithm assumes all images are of same person
4. **Expression Mixing**: Averaging expressions can create unrealistic results
5. **Memory**: Loading multiple images requires more GPU memory

## Troubleshooting

### Poor Results
- Check if images are of same person
- Verify face detection worked on all images
- Try different merging strategies
- Reduce number of images (quality > quantity)

### Inconsistent Geometry
- Increase weight on frontal, neutral images
- Use outlier removal in averaging
- Check for failed face detections

### Memory Issues
- Process images in smaller batches
- Reduce image resolution
- Use CPU for encoding, GPU for decoding

## References

- Original DECA paper: [Learning an Animatable Detailed 3D Face Model](https://arxiv.org/abs/2012.04012)
- DECA GitHub: https://github.com/yfeng95/DECA
- FLAME model: http://flame.is.tue.mpg.de/
