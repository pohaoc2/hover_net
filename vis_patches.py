import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import glob
import os
import re

def get_class_colors():
    """Return color map for each nucleus class."""
    colors = {
        0: (0, 0, 0),           # background: black
        1: (255, 255, 0),       # other: yellow
        2: (255, 0, 255),       # inflammatory: pink
        3: (0, 255, 0),         # healthy epithelial: green
        4: (255, 0, 0),         # dysplastic/malignant: red
        5: (0, 0, 255),         # fibroblast: blue
        6: (0, 255, 255),       # muscle: cyan
        7: (244, 158, 66),      # endothelial: orange
    }
    return colors

def visualize_patch_with_overlay(npy_path, contour_thickness=2, figsize=(16, 8), save_path=None):
    """
    Load and visualize patch with nucleus contours and centroids overlaid.
    
    Args:
        npy_path: path to the .npy file
        contour_thickness: thickness of contour lines
    """
    # Load the patch
    patch = np.load(npy_path)
    
    # Split channels
    img = patch[..., :3].astype(np.uint8)  # RGB image
    inst_map = patch[..., 3].astype(np.int32)  # Instance map
    type_map = patch[..., 4].astype(np.int32)  # Type map
    
    print(f"Patch shape: {patch.shape}")
    print(f"Number of unique nuclei: {len(np.unique(inst_map)) - 1}")  # -1 for background
    
    # Create overlay image
    overlay = img.copy()
    
    # Get color map
    colors = get_class_colors()
    
    # Get unique nucleus IDs (excluding background 0)
    nucleus_ids = np.unique(inst_map)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    
    centroids = []
    contour_info = []
    
    for nucleus_id in nucleus_ids:
        # Create binary mask for this nucleus
        mask = (inst_map == nucleus_id).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        # Get the nucleus type from type_map
        nucleus_pixels = type_map[inst_map == nucleus_id]
        nucleus_type = np.bincount(nucleus_pixels).argmax()  # Most common type
        
        # Get color for this type
        color = colors.get(nucleus_type, (255, 255, 255))
        
        # Draw contours
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)
        
        # Calculate centroid
        M = cv2.moments(contours[0])
        if len(contours) > 1:
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy, nucleus_type))
                    cv2.circle(overlay, (cx, cy), 3, color, -1)
                    cv2.circle(overlay, (cx, cy), 4, (255, 255, 255), 1)  # White outline
        else:
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy, nucleus_type))
                
                # Draw centroid
                cv2.circle(overlay, (cx, cy), 3, color, -1)
                cv2.circle(overlay, (cx, cy), 4, (255, 255, 255), 1)  # White outline
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Overlay image with contours and centroids
    axes[1].imshow(overlay)
    axes[1].set_title(f'With Nucleus Contours & Centroids\n({len(centroids)} nuclei)', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(colors[1])/255, label='Other'),
        Patch(facecolor=np.array(colors[2])/255, label='Inflammatory'),
        Patch(facecolor=np.array(colors[3])/255, label='Healthy Epithelial'),
        Patch(facecolor=np.array(colors[4])/255, label='Dysplastic/Malignant'),
        Patch(facecolor=np.array(colors[5])/255, label='Fibroblast'),
        Patch(facecolor=np.array(colors[6])/255, label='Muscle'),
        Patch(facecolor=np.array(colors[7])/255, label='Endothelial'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    return centroids


def visualize_multiple_patches_with_overlay(patch_dir, num_samples=4, contour_thickness=2, figsize=(16, 8)):
    """
    Visualize multiple patches with overlays in a grid.
    
    Args:
        patch_dir: directory containing .npy files
        num_samples: number of patches to display
        contour_thickness: thickness of contour lines
    """
    import glob
    
    # Get all .npy files
    npy_files = glob.glob(f"{patch_dir}/*.npy")
    npy_files.sort()
    
    num_samples = min(num_samples, len(npy_files))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    colors = get_class_colors()
    
    for i in range(num_samples):
        patch = np.load(npy_files[i])
        
        img = patch[..., :3].astype(np.uint8)
        inst_map = patch[..., 3].astype(np.int32)
        type_map = patch[..., 4].astype(np.int32)
        
        overlay = img.copy()
        
        # Get unique nucleus IDs
        nucleus_ids = np.unique(inst_map)
        nucleus_ids = nucleus_ids[nucleus_ids != 0]
        
        for nucleus_id in nucleus_ids:
            mask = (inst_map == nucleus_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            nucleus_pixels = type_map[inst_map == nucleus_id]
            nucleus_type = np.bincount(nucleus_pixels).argmax()
            color = colors.get(nucleus_type, (255, 255, 255))
            
            cv2.drawContours(overlay, contours, -1, color, contour_thickness)
            
            # Calculate and draw centroid
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(overlay, (cx, cy), 3, color, -1)
                cv2.circle(overlay, (cx, cy), 4, (255, 255, 255), 1)
        
        # Plot original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Patch {i} - Original', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Plot overlay
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f'Patch {i} - With Annotations ({len(nucleus_ids)} nuclei)', 
                            fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{patch_dir}/multiple_patches_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Visualize all patches in the folder
    path_dir = "dataset/training_data/consep/consep/train/540x540_164x164/"
    
    # Get all .npy files matching the pattern train_*_*.npy
    npy_files = glob.glob(os.path.join(path_dir, "train_*_*.npy"))
    
    # Sort files by idx first, then by number (train_{idx}_{number}.npy)
    def extract_sort_key(filename):
        match = re.search(r'train_(\d+)_(\d+)\.npy', os.path.basename(filename))
        if match:
            idx = int(match.group(1))
            number = int(match.group(2))
            return (idx, number)
        return (0, 0)
    
    npy_files.sort(key=extract_sort_key)
    
    print(f"Found {len(npy_files)} .npy files to process")
    
    for patch_path in npy_files:
        # Save patch as image
        patch = np.load(patch_path)
        img = patch[..., :3].astype(np.uint8)
        cv2.imwrite(patch_path.replace(".npy", ".png"), img)
        print(f"Saved {patch_path} as {patch_path.replace('.npy', '.png')}")
    asd()

    # Iterate over all .npy files
    for patch_path in npy_files:
        # Extract the base filename without extension
        base_name = os.path.splitext(os.path.basename(patch_path))[0]
        save_path = os.path.join(path_dir, f"{base_name}_visualization.png")
        
        print(f"Processing: {os.path.basename(patch_path)}")
        
        centroids = visualize_patch_with_overlay(
            patch_path,
            contour_thickness=2,
            save_path=save_path
        )
    visualize_multiple_patches_with_overlay(
        path_dir, 
        num_samples=2, 
        contour_thickness=2,
        figsize=(12, 9)
    )