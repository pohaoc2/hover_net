import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

HEX_SIZE = 20#30 / 3 **0.5
IMAGE_SIZE = 540
CIRCULARITY_MIN = 0.7
CIRCULARITY_MAX = 1
OVERLAP_OFFSET_FACTOR = 0.4
RANDOM_SEED = 42
MPP = 0.42


def hex_to_xy(u, v, w, hex_size, center_x, center_y):
    """Convert hexagonal (cube) coordinates to Cartesian coordinates."""
    q = u
    r = v
    
    x_rel = hex_size * (3.0/2.0 * q)
    y_rel = hex_size * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
    
    x = x_rel + center_x
    y = y_rel + center_y
    
    return x, y

def calculate_ellipse_axes(volume, height, circularity_min, circularity_max, mpp=1.0):
    """
    Calculate ellipse semi-axes (a, b) from volume and height.
    Volume = π * a * b * height
    
    Args:
        volume: Cell volume in um^3
        height: Cell height in um
        circularity_min: Minimum circularity
        circularity_max: Maximum circularity
        mpp: Microns per pixel - conversion factor from micrometers to pixels
    
    Returns:
        a, b: Ellipse semi-axes in pixels
    """
    circularity = np.random.uniform(circularity_min, circularity_max)
    # Volume and height are in micrometers, so area is in um^2
    area_um2 = volume / height
    # Calculate axes in micrometers
    a_um = np.sqrt(area_um2 / (np.pi * circularity))
    b_um = circularity * a_um
    # Convert from micrometers to pixels
    a = a_um / mpp
    b = b_um / mpp
    return a, b

def get_cell_position(cell_id, hex_coord, id_to_location, locations_data, 
                     hex_size, center_x, center_y, overlap_offset_factor):
    """Calculate cell position with hexagonal grid offset for overlapping cells."""
    u, v, w, z = hex_coord
    base_x, base_y = hex_to_xy(u, v, w, hex_size, center_x, center_y)
    
    # Find cells at this location
    cells_at_location = []
    for loc in locations_data:
        if loc['coordinate'] == hex_coord:
            cells_at_location = loc['ids']
            break
    
    # Add offset if multiple cells at same location
    if len(cells_at_location) > 1:
        # Find index of current cell
        cell_index = cells_at_location.index(cell_id)
        
        # Place cells in hexagonal pattern clockwise starting from right
        # Hexagonal directions (flat-top): 0°, 60°, 120°, 180°, 240°, 300°
        if cell_index == 0:
            # Center cell
            offset_x, offset_y = 0, 0
        else:
            # Surrounding cells in clockwise order
            angle_deg = [0, 60, 120, 180, 240, 300][cell_index - 1] if cell_index <= 6 else (cell_index - 1) * 60
            angle_rad = np.radians(angle_deg)
            offset_distance = hex_size * overlap_offset_factor * 2  # Distance from center
            offset_x = offset_distance * np.cos(angle_rad)
            offset_y = -offset_distance * np.sin(angle_rad)  # Negative because y-axis is inverted
    else:
        offset_x = offset_y = 0
    
    return base_x + offset_x, base_y + offset_y

def draw_binary_mask_to_array(cells_data, id_to_location, locations_data,
                              hex_size, img_size, center_x, center_y,
                              circularity_min, circularity_max, overlap_offset_factor,
                              mpp=1.0):
    """
    Draw binary mask directly as numpy array - most efficient.
    
    Args:
        mpp: Microns per pixel - conversion factor from micrometers to pixels
             (e.g., mpp=0.5 means 1 pixel = 0.5 microns)
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for cell in cells_data:
        cell_id = cell['id']
        volume = cell['volume']  # in um^3
        height = cell['height']  # in um
        hex_coord = id_to_location[cell_id]
        
        x, y = get_cell_position(cell_id, hex_coord, id_to_location,
                                locations_data, hex_size, center_x, center_y,
                                overlap_offset_factor)
        
        a, b = calculate_ellipse_axes(volume, height, circularity_min, circularity_max, mpp)
        angle = np.random.uniform(0, 360)
        
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        y_min = max(0, int(y - max(a, b) - 1))
        y_max = min(img_size, int(y + max(a, b) + 1))
        x_min = max(0, int(x - max(a, b) - 1))
        x_max = min(img_size, int(x + max(a, b) + 1))
        
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                dx = px - x
                dy = py - y
                dx_rot = dx * cos_a + dy * sin_a
                dy_rot = -dx * sin_a + dy * cos_a
                
                if (dx_rot/a)**2 + (dy_rot/b)**2 <= 1:
                    mask[py, px] = cell_id
    plt.imshow(mask, cmap='jet')
    plt.savefig('./mask.png')
    #asd()
    return mask

def create_id_to_location_mapping(locations_data):
    """Create mapping from cell ID to location."""
    id_to_location = {}
    for loc in locations_data:
        coord = loc['coordinate']
        cell_ids = loc['ids']
        for cell_id in cell_ids:
            id_to_location[cell_id] = coord
    return id_to_location


def process_single_mask(args):
    """
    Worker function to process a single mask file.
    This function is designed to be called in parallel.
    
    Args:
        args: Tuple of (file_paths, output_path, hex_size, img_size, 
                       circularity_min, circularity_max, overlap_offset_factor, random_seed, mpp)
    
    Returns:
        Tuple of (output_path, success, error_message)
    """
    file_paths, output_path, hex_size, img_size, circularity_min, circularity_max, overlap_offset_factor, random_seed, mpp = args
    try:
        with open(file_paths[0], 'r') as f:
            cells_data = json.load(f)
        
        with open(file_paths[1], 'r') as f:
            locations_data = json.load(f)
        
        # Derived parameters
        center_x = center_y = img_size // 2
        
        # Create mapping
        id_to_location = create_id_to_location_mapping(locations_data)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Generate mask
        mask = draw_binary_mask_to_array(cells_data, id_to_location, locations_data,
                        hex_size, img_size, center_x, center_y,
                        circularity_min, circularity_max, overlap_offset_factor, mpp)
        
        # Save mask
        cv2.imwrite(output_path, mask)
        
        return (output_path, True, None)
    except Exception as e:
        return (output_path, False, str(e))


def main():
    # Load data
    process_args = []
    if 1:
        folder_path = 'dataset/training_data/consep/consep/train/540x540_164x164/mask_original/'
        # Prepare arguments for parallel processing
        
        for number in range(1, 28):
            for sub_number in range(49):
                cell_path = folder_path + f'train_{number}_{sub_number}_0000_000000.CELLS.json'
                location_path = folder_path + f'train_{number}_{sub_number}_0000_000000.LOCATIONS.json'
                output_path = folder_path + f'train_{number}_{sub_number}.mask.png'
                process_args.append((
                    [cell_path, location_path], output_path, HEX_SIZE, IMAGE_SIZE,
                    CIRCULARITY_MIN, CIRCULARITY_MAX, OVERLAP_OFFSET_FACTOR, RANDOM_SEED, MPP
                ))
    if 0:
        folder_path = '../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/iter_0/inputs/'
        for input_id in range(1, 1025):
            cell_path = folder_path + f"input_{input_id}/combined_grid_0009_010080.CELLS.json"
            location_path = folder_path + f"input_{input_id}/combined_grid_0009_010080.LOCATIONS.json"
            os.makedirs(folder_path + f"mask_pngs", exist_ok=True)
            output_path = folder_path + f"mask_pngs/input_{input_id}.mask.png"
            process_args.append((
                [cell_path, location_path], output_path, HEX_SIZE, IMAGE_SIZE,
                CIRCULARITY_MIN, CIRCULARITY_MAX, OVERLAP_OFFSET_FACTOR, RANDOM_SEED, MPP
            ))
    process_args = process_args[:]
    # Process files in parallel
    max_workers = int(os.cpu_count() / 2) or 4
    print(f"Processing {len(process_args)} mask files in parallel using {max_workers} workers...")
    
    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_mask, args): args for args in process_args}
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_args):
            output_path, success, error = future.result()
            if success:
                completed += 1
                print(f"[{completed}/{len(process_args)}] Completed: {output_path}")
            else:
                failed += 1
                print(f"[ERROR] Failed to process {output_path}: {error}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {completed}/{len(process_args)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(process_args)}")


if __name__ == '__main__':
    main()