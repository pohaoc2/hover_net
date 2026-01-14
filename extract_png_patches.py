import math
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

WIN_SIZE = [540, 540]
STEP_SIZE = [164, 164]
EXTRACT_TYPE = "mirror"
NUM_WORKERS = os.cpu_count() or 4
original_image_width = 2048 # in pixels
original_image_height = 1536 # in pixels
micron_per_pixel = 0.42
DESIRED_MPP = 0.5  # Set to None to use original resolution, or specify desired microns per pixel (e.g., 0.42)


def extract_patches_with_mirror(image, win_size, step_size):
    """Extract patches from image with mirror padding.
    
    Args:
        image: Input image as numpy array (H, W, C)
        win_size: Window size [height, width]
        step_size: Step size [height, width]
    
    Returns:
        List of patches
    """
    im_h, im_w = image.shape[:2]
    win_h, win_w = win_size
    step_h, step_w = step_size
    
    # Calculate padding needed for mirror extraction
    diff_h = win_h - step_h
    padt = diff_h // 2
    padb = diff_h - padt
    
    diff_w = win_w - step_w
    padl = diff_w // 2
    padr = diff_w - padl
    
    # Apply mirror padding (reflect mode)
    if len(image.shape) == 2:
        # Grayscale image
        padded_image = np.pad(image, ((padt, padb), (padl, padr)), mode='reflect')
    else:
        # Color image
        padded_image = np.pad(image, ((padt, padb), (padl, padr), (0, 0)), mode='reflect')
    
    # Calculate number of patches
    padded_h, padded_w = padded_image.shape[:2]
    
    def extract_infos(length, win_size, step_size):
        flag = (length - win_size) % step_size != 0
        last_step = math.floor((length - win_size) / step_size)
        last_step = (last_step + 1) * step_size
        return flag, last_step
    
    h_flag, h_last = extract_infos(padded_h, win_h, step_h)
    w_flag, w_last = extract_infos(padded_w, win_w, step_w)
    
    patches = []
    
    # Extract valid patches
    for row in range(0, h_last, step_h):
        for col in range(0, w_last, step_w):
            patch = padded_image[row:row+win_h, col:col+win_w]
            patches.append(patch)
    
    # Handle edge cases
    if h_flag:
        row = padded_h - win_h
        for col in range(0, w_last, step_w):
            patch = padded_image[row:row+win_h, col:col+win_w]
            patches.append(patch)
    
    if w_flag:
        col = padded_w - win_w
        for row in range(0, h_last, step_h):
            patch = padded_image[row:row+win_h, col:col+win_w]
            patches.append(patch)
    
    if h_flag and w_flag:
        patch = padded_image[padded_h-win_h:padded_h, padded_w-win_w:padded_w]
        patches.append(patch)
    
    return patches


def process_single_image(image_path, output_folder, win_size, step_size, original_mpp, desired_mpp):
    """Process a single image and save its patches.
    
    Args:
        image_path: Path to input image
        output_folder: Path to output folder
        win_size: Window size [height, width]
        step_size: Step size [height, width]
        original_mpp: Original microns per pixel of the image
        desired_mpp: Desired microns per pixel (None to use original resolution)
    
    Returns:
        Number of patches extracted
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return 0
        
        # Convert BGR to RGB if needed (cv2 loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image if desired MPP is specified and different from original
        if desired_mpp is not None and desired_mpp != original_mpp:
            scale_factor = original_mpp / desired_mpp
            h, w = image.shape[:2]
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Extract patches
        patches = extract_patches_with_mirror(image, win_size, step_size)
        
        # Save patches
        image_stem = image_path.stem
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for idx, patch in enumerate(patches):
            # Convert back to BGR for saving with cv2
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            patch_path = output_folder / f"{image_stem}_{idx:04d}.png"
            cv2.imwrite(str(patch_path), patch_bgr)
        
        return len(patches)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0


def main():
    image_folder = Path("../share_space/data/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/Invasive_PNG")
    output_folder = image_folder.parent / f"{image_folder.name}_patches"
    
    if not image_folder.exists():
        print(f"Error: Input folder {image_folder} does not exist")
        return
    
    # Find all PNG images
    image_files = list(image_folder.glob("*.png"))
    if len(image_files) == 0:
        print(f"Warning: No PNG images found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} PNG images")
    print(f"Window size: {WIN_SIZE}, Step size: {STEP_SIZE}")
    print(f"Original MPP: {micron_per_pixel}, Desired MPP: {DESIRED_MPP if DESIRED_MPP is not None else 'original'}")
    print(f"Using {NUM_WORKERS} workers")
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel
    total_patches = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, img_path, output_folder, 
                          WIN_SIZE, STEP_SIZE, micron_per_pixel, DESIRED_MPP): img_path
            for img_path in image_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for future in as_completed(future_to_image):
                img_path = future_to_image[future]
                try:
                    num_patches = future.result()
                    total_patches += num_patches
                    pbar.update(1)
                    pbar.set_postfix({'patches': total_patches})
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
                    pbar.update(1)
    
    print(f"\nCompleted! Extracted {total_patches} patches total")
    print(f"Patches saved to: {output_folder}")


if __name__ == "__main__":
    main()