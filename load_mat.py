import os
import numpy as np
import cv2
from scipy.io import loadmat
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def overlay_nuclei_segmentation(
    image_path: str,
    mat_path: str,
    alpha: float = 0.7,
    contour_thickness: int = 2
) -> np.ndarray:
    """
    Overlay nucleus segmentation contours on the original image with color-coded types.
    
    Parameters:
    -----------
    image_path : str
        Path to the original image file
    mat_path : str
        Path to the .mat file containing segmentation ground truth
    alpha : float
        Transparency for overlay (0.0 = fully transparent, 1.0 = fully opaque)
    contour_thickness : int
        Thickness of contour lines in pixels
        
    Returns:
    --------
    np.ndarray
        Image with overlaid contours
    """
    
    # Define color map for each nucleus type (RGB format)
    color_map = {
        1: (255, 255, 0), # other is yellow
        2: (255, 0, 255), # inflammatory is pink
        3: (0, 255, 0), # healthy epithelial is green
        4: (255, 0, 0), # dysplastic/malignant epithelial is red
        5: (0, 0, 255), # fibroblast is blue
        6: (0, 255, 255), # muscle is cyan
        7: (244, 158, 66) # endothelial is orange
    }
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the .mat file
    mat_data = loadmat(mat_path)
    inst_map = mat_data['inst_map']
    inst_type = np.array(mat_data['inst_type'].flatten())
    print(f"inst_type = {np.unique(inst_type)}")
    # Create overlay image
    overlay = image_rgb.copy()
    
    # Get unique nucleus IDs (excluding background)
    nucleus_ids = np.unique(inst_map)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    
    # Process each nucleus
    for nucleus_id in nucleus_ids:
        # Create binary mask for this nucleus
        nucleus_id = int(nucleus_id)
        nucleus_mask = (inst_map == nucleus_id).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            nucleus_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        nucleus_type = int(inst_type[nucleus_id - 1])
        
        # Get color for this type (RGB format)
        color_rgb = color_map.get(nucleus_type, (255, 255, 255))
        # Draw contours on overlay (overlay is already in RGB format)
        cv2.drawContours(
            overlay, 
            contours, 
            -1, 
            color_rgb,  # Use RGB color directly since overlay is RGB
            contour_thickness
        )
    
    # Blend original image with overlay
    result = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    
    return result


def visualize_with_legend(
    result_image: np.ndarray,
    true_image: np.ndarray,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Display the result with a legend showing nucleus types and colors.
    
    Parameters:
    -----------
    result_image : np.ndarray
        The overlaid image to display
    figsize : Tuple[int, int]
        Figure size for matplotlib
    """
    
    # Define labels and colors for legend (RGB format)
    labels_colors = {
        'Other': (255/255, 255/255, 0/255), # yellow
        'Inflammatory': (255/255, 0/255, 255/255), # pink
        'Healthy Epithelial': (0/255, 255/255, 0/255), # green      
        'Dysplastic/Malignant Epithelial': (255/255, 0/255, 0/255), # red
        'Fibroblast': (0/255, 0/255, 255/255), # blue
        'Muscle': (0/255, 255/255, 255/255), # cyan
        'Endothelial': (244/255, 158/255, 66/255) # orange
    }    
    
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(result_image)
    ax[0].axis('off')
    ax[0].set_title('Nuclei Segmentation Overlay', fontsize=16, pad=20)
    ax[1].imshow(true_image)
    ax[1].axis('off')
    ax[1].set_title('True Overlay', fontsize=16, pad=20)
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=label)
        for label, color in labels_colors.items()
    ]
    ax[1].legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.15, 1),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.show()

GT_tile_path = '/home/pohaoc2/UW/bagherilab/hover_net/dataset/CoNSeP/Train/Images/'
GT_mat_path = '/home/pohaoc2/UW/bagherilab/hover_net/dataset/CoNSeP/Train/Labels/'
GT_true_overlap_path = '/home/pohaoc2/UW/bagherilab/hover_net/dataset/CoNSeP/Train/Overlay/'
# Create overlay
for index in range(1, 10):
    print(f"Processing train_{index}.png")
    result = overlay_nuclei_segmentation(
        image_path=GT_tile_path + "train_"+str(index)+".png",
        mat_path=GT_mat_path + "train_"+str(index)+".mat",
        alpha=1.0,
        contour_thickness=2
    )

    gt_true_image_path = os.path.join(GT_true_overlap_path, "train_"+str(index)+".png")
    if not os.path.exists(gt_true_image_path):
        os.path.dirname(GT_true_overlap_path) and print(f"Files in {GT_true_overlap_path}:", os.listdir(GT_true_overlap_path))
        raise FileNotFoundError(f"Could not find image at path: {gt_true_image_path}")
    #read the image as RGB
    gt_true_image = cv2.cvtColor(cv2.imread(gt_true_image_path), cv2.COLOR_BGR2RGB)
    if gt_true_image is None:
        raise ValueError(f"Failed to load image at path: {gt_true_image_path}")

    gt_true_image = np.array(gt_true_image)
    print(f"Result image shape: {result.shape}")
    print(f"Loaded image shape: {gt_true_image.shape}")
    # Visualize with legend
    visualize_with_legend(result, gt_true_image)

# Optionally save the result
# plt.imsave('overlay_result.png', result)

tile_path = './dataset/sample_tiles/imgs/'
tile_mat_path = './dataset/sample_tiles/pred/mat/'
tile_overlay_path = './dataset/sample_tiles/pred/overlay/'