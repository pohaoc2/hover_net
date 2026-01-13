import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing compatibility
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import cv2
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
type_to_name = {
    1: "OTHER",
    2: "INFLAMMATORY",
    3: "HEALTHY_EPITHELIAL",
    4: "DYSPLASTIC/MALIGNANT",
    5: "FIBROBLAST",
    6: "MUSCLE",
    7: "ENDOTHELIAL",
}

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

def extract_centroids_from_patch(npy_path):
    """
    Extract centroids and cell properties from a patch.
    
    Returns:
        list of dict: Each dict contains centroid info and cell properties
        img: Original image
    """
    patch = np.load(npy_path)
    
    img = patch[..., :3].astype(np.uint8)
    inst_map = patch[..., 3].astype(np.int32)
    type_map = patch[..., 4].astype(np.int32)
    nucleus_ids = np.unique(inst_map)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    cells = []
    mirror_cell_count = 0
    for nucleus_id in nucleus_ids:
        mask = (inst_map == nucleus_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Get nucleus type
        nucleus_pixels = type_map[inst_map == nucleus_id]
        nucleus_type = np.bincount(nucleus_pixels).argmax()
        
        # Calculate centroid
        try:
            if len(contours) > 1:
                for contour in contours:
                    mirror_cell_count += 1
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                    area = cv2.contourArea(contour)
                    cells.append({
                        'id': len(nucleus_ids) + mirror_cell_count,
                        'x': cx,
                        'y': cy,
                        'type': int(nucleus_type),
                        'area': area
                    })
            else:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    # Calculate area (volume in 2D)
                    area = cv2.contourArea(contours[0])
                    
                    cells.append({
                        'id': int(nucleus_id),
                        'x': cx,
                        'y': cy,
                        'type': int(nucleus_type),
                        'area': area
                    })
        except Exception as e:
            print(f"Error extracting centroids from patch: {e}")
            print(f"Nucleus ID: {nucleus_id}")
            print(f"Contours: {contours}")
            print(f"Nucleus type: {nucleus_type}")
            print(f"Nucleus pixels: {nucleus_pixels}")
            print(f"Nucleus mask: {mask}")
            print(f"Nucleus inst_map: {inst_map}")
            print(f"Nucleus type_map: {type_map}")
    return cells, img

def xy_to_hex(x, y, hex_size, center_x, center_y):
    """
    Convert Cartesian coordinates to hexagonal (cube) coordinates.
    Using flat-top hexagon orientation with (0,0,0) at the center of the image.
    
    Args:
        x, y: Cartesian coordinates in pixels
        hex_size: Size of hexagon in pixels (distance from center to vertex)
        center_x, center_y: Center of the image (origin for hexagonal coordinates)
    
    Returns:
        (u, v, w): Cube coordinates where u + v + w = 0
    """
    # Offset coordinates relative to center
    x_rel = x - center_x
    y_rel = y - center_y
    
    # Convert to axial coordinates (q, r)
    q = (2.0/3.0 * x_rel) / hex_size
    r = (-1.0/3.0 * x_rel + np.sqrt(3)/3 * y_rel) / hex_size
    
    # Round to nearest hexagon
    q_round = np.round(q)
    r_round = np.round(r)
    
    # Convert axial to cube coordinates
    u = q_round
    v = r_round
    w = -q_round - r_round
    
    # Handle rounding errors (ensure u + v + w = 0)
    q_diff = abs(q - q_round)
    r_diff = abs(r - r_round)
    s_diff = abs(-q - r - (-q_round - r_round))
    
    if q_diff > r_diff and q_diff > s_diff:
        u = -v - w
    elif r_diff > s_diff:
        v = -u - w
    else:
        w = -u - v
    
    return int(u), int(v), int(w)

def hex_to_xy(u, v, w, hex_size, center_x, center_y):
    """
    Convert hexagonal (cube) coordinates back to Cartesian coordinates.
    
    Args:
        u, v, w: Cube coordinates
        hex_size: Size of hexagon in pixels
        center_x, center_y: Center of the image (origin for hexagonal coordinates)
    
    Returns:
        (x, y): Cartesian coordinates
    """
    q = u
    r = v
    
    x_rel = hex_size * (3.0/2.0 * q)
    y_rel = hex_size * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
    
    # Add back the center offset
    x = x_rel + center_x
    y = y_rel + center_y
    
    return x, y

def convert_to_hexagonal_system(cells, img_shape, hex_size, idx, number, save_dir):
    """
    Convert cells to hexagonal coordinate system and save output files.
    
    Args:
        cells: List of cell dictionaries with x, y coordinates
        img_shape: Shape of the image (height, width)
        hex_size: Hexagon size in pixels
        output_prefix: Prefix for output files (e.g., 'patch_001')
    
    Returns:
        locations_data, cells_data, center: The data structures that were saved and the center coordinates
    """
    # Calculate image center
    height, width = img_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    
    # Group cells by hexagonal coordinate
    hex_cells = defaultdict(list)
    
    for cell in cells:
        u, v, w = xy_to_hex(cell['x'], cell['y'], hex_size, center_x, center_y)
        hex_cells[(u, v, w)].append(cell)
    
    # Create LOCATIONS data
    locations_data = []
    
    for (u, v, w), cell_list in hex_cells.items():
        cell_ids = [cell['id'] for cell in cell_list]
        locations_data.append({
            "coordinate": [u, v, w, 0],  # 4th coordinate is z-level (0 for 2D)
            "ids": cell_ids
        })
    
    # Create CELLS data
    cells_data = []

    for (u, v, w), cell_list in hex_cells.items():
        for cell in cell_list:            
            cells_data.append({
                "id": cell['id'],
                "parent": 0,
                "pop": 1,  # Using nucleus type as population
                "age": 0,
                "divisions": 50,
                "state": type_to_name[cell['type']],
                "volume": float(cell['area']),  # Using area as volume
                "height": 8.7,  # Default height (you can adjust this)
                "criticals": [float(cell['area']), 13.78],
                "cycles": []
            })
    

    # Sort by ID
    cells_data.sort(key=lambda x: x['id'])
    
    # Save to JSON files
    locations_file = f"{save_dir}train_{idx}_{number}_0000_000000.LOCATIONS.json"
    cells_file = f"{save_dir}train_{idx}_{number}_0000_000000.CELLS.json"
    
    with open(locations_file, 'w') as f:
        f.write('[\n')
        for i, loc in enumerate(locations_data):
            f.write('  {\n')
            f.write(f'    "coordinate": {json.dumps(loc["coordinate"])},\n')
            f.write(f'    "ids": {json.dumps(loc["ids"])}\n')
            if i < len(locations_data) - 1:
                f.write('  },\n')
            else:
                f.write('  }\n')
        f.write(']\n')

    with open(cells_file, 'w') as f:
        json.dump(cells_data, f, indent=2)
    
    print(f"Saved {len(locations_data)} hexagonal locations to {locations_file}")
    print(f"Saved {len(cells_data)} cells to {cells_file}")
    print(f"Image center (0,0,0) at pixel coordinates: ({center_x:.1f}, {center_y:.1f})")
    
    return locations_data, cells_data, (center_x, center_y)

def plot_hexagonal_population(locations_data, hex_size, center_x, center_y, target_size, max_cells, output_path=None, use_white_background=True, target_mpp=0.5):
    """
    Plot the hexagonal grid with grayscale fill indicating cell counts.

    Args:
        locations_data: List of hexagonal locations, each with coordinate and ids.
        hex_size: Size of hexagon in microns.
        center_x, center_y: Center of the grid in pixel coordinates.
        target_size: Size of the output image in pixels (assumed square).
        max_cells: Maximum number of cells in any hexagon (for normalization).
        output_path: Path to save the plot. If None, does not save.
        use_white_background: If True, white background; else black.
        target_mpp: Target microns per pixel (default 0.5).
    """
    from matplotlib.patches import Polygon

    # Convert hex_size from microns to pixels
    hex_size_pixels = hex_size / target_mpp
    
    # Calculate figure size and DPI to get exact pixel dimensions
    # Using figsize in inches and dpi to control output size
    dpi = 100
    figsize_inches = target_size / dpi
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(figsize_inches, figsize_inches))
    ax1.set_xlim(0, target_size)
    ax1.set_ylim(target_size, 0)
    ax1.axis('off')
    fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Set background color based on use_white_background
    bg_color = 'white' if use_white_background else 'black'
    
    for location in locations_data:
        u, v, w, _ = location['coordinate']
        x, y = hex_to_xy(u, v, w, hex_size_pixels, center_x, center_y)
        
        # Calculate grayscale value based on number of cells
        num_cells_in_hex = len(location['ids'])
        if use_white_background:
            # White background: more cells = darker (closer to 0/black)
            # Normalize: 0 cells = white (1.0), max_cells = black (0.0)
            gray_value = 1.0 - (num_cells_in_hex / max_cells) if max_cells > 0 else 1.0
            hex_path = output_path.replace('_hexagonal_visualization.png', '_0000.000000.population.count.white_bg.png') if output_path else None
        else:
            # Black background: more cells = lighter (closer to 1.0/white)
            # Normalize: 0 cells = black (0.0), max_cells = white (1.0)
            gray_value = (num_cells_in_hex / max_cells) if max_cells > 0 else 0.0
            hex_path = output_path.replace('_hexagonal_visualization.png', '_0000.000000.population.count.black_bg.png') if output_path else None
            
        # Create hexagon vertices
        angles = np.linspace(0, 2*np.pi, 7)
        hex_x = x + hex_size_pixels * np.cos(angles)
        hex_y = y + hex_size_pixels * np.sin(angles)
        hex_vertices = np.column_stack([hex_x, hex_y])
        
        # Fill hexagon with grayscale color
        hex_polygon = Polygon(hex_vertices, closed=True, 
                            facecolor=(gray_value, gray_value, gray_value), 
                            edgecolor='k',
                            linewidth=2.5,
                            alpha=1.0)
        ax1.add_patch(hex_polygon)
    
    if hex_path:
        os.makedirs(os.path.dirname(hex_path), exist_ok=True)
        plt.savefig(hex_path, dpi=dpi, pad_inches=0, facecolor=bg_color)
    else:
        plt.show()
    plt.close(fig1)


def visualize_hexagonal_conversion(img, cells, hex_size, locations_data, center, output_path=None, use_white_background=True):
    """
    Visualize the conversion from Cartesian to hexagonal coordinates.
    Creates two separate 540x540 images:
    1. Hexagonal grid with grayscale fill (no original image)
    2. Original image masked to only show hexagonal regions
    
    Args:
        use_white_background: If True, background is white and more cells = darker.
                             If False, background is black and more cells = lighter.
    """
    from matplotlib.patches import Polygon
    
    center_x, center_y = center
    height, width = img.shape[:2]
    
    # Ensure image is 540x540
    target_size = 540
    if height != target_size or width != target_size:
        img_resized = cv2.resize(img, (target_size, target_size))
    else:
        img_resized = img.copy()
    
    # Find maximum number of cells in any hexagon for normalization
    max_cells = max(len(location['ids']) for location in locations_data) if locations_data else 1

    # Call the extracted function in place of the original code
    plot_hexagonal_population(
        locations_data, hex_size, center_x, center_y,
        target_size=target_size, max_cells=max_cells,
        output_path=output_path, use_white_background=use_white_background
    )
    
    # Image 2: Original image masked to only show hexagonal regions
    fig2, ax2 = plt.subplots(1, 1, figsize=(target_size/100, target_size/100))
    
    # Create a black background
    masked_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Create combined mask for all hexagons
    combined_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    
    for location in locations_data:
        u, v, w, _ = location['coordinate']
        x, y = hex_to_xy(u, v, w, hex_size, center_x, center_y)
        
        # Create hexagon vertices
        angles = np.linspace(0, 2*np.pi, 7)
        hex_x = x + hex_size * np.cos(angles)
        hex_y = y + hex_size * np.sin(angles)
        hex_points = np.array([(int(px), int(py)) for px, py in zip(hex_x, hex_y)], dtype=np.int32)
        
        # Add to combined mask
        cv2.fillPoly(combined_mask, [hex_points], 255)
    
    # Apply mask to original image
    masked_img = cv2.bitwise_and(img_resized, img_resized, mask=combined_mask)
    
    ax2.imshow(masked_img)
    ax2.axis('off')
    ax2.set_xlim(0, target_size)
    ax2.set_ylim(target_size, 0)
    
    masked_path = output_path.replace('_hexagonal_visualization.png', '_masked_original.png') if output_path else None
    if masked_path:
        plt.savefig(masked_path, dpi=100, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close(fig2)

# Main conversion function
def convert_patch_to_hexagonal(npy_path, hex_size, idx, number, save_dir):
    """
    Main function to convert a patch from Cartesian to hexagonal system.
    
    Args:
        npy_path: Path to the .npy patch file
        hex_size: Hexagon size in pixels
        output_prefix: Prefix for output files (default: uses npy filename)
    
    Returns:
        cells, locations_data, cells_data
    """
    print(f"Processing: {npy_path}")
    print(f"Hexagon size: {hex_size} pixels")
    print("-" * 60)
    
    # Extract centroids and cell properties
    cells, img = extract_centroids_from_patch(npy_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis('off')
    plt.savefig(f"{save_dir}sandbox_{idx}_{number}_original.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Extracted {len(cells)} cells")
    
    # Convert to hexagonal system
    locations_data, cells_data, center = convert_to_hexagonal_system(
        cells, img.shape, hex_size, idx, number, save_dir   
    )
    
    # Visualize
    visualize_hexagonal_conversion(img, cells, hex_size, locations_data, center,
                                   output_path=f"{save_dir}train_{idx}_{number}_hexagonal_visualization.png",
                                   use_white_background=False)
    
    return cells, locations_data, cells_data

def save_binary_nuclei_map(npy_path, save_dir, idx, number, with_contour=False):
    patch = np.load(npy_path)
    inst_map = patch[..., 3].astype(np.int32)
    mask = (inst_map > 0).astype(np.uint8)
    
    if with_contour:
        # Get unique instance IDs (excluding background)
        instance_ids = np.unique(inst_map)
        instance_ids = instance_ids[instance_ids != 0]
        
        # Draw contours for each instance to split them
        for instance_id in instance_ids:
            instance_mask = (inst_map == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours as black (0) to create boundaries between instances
            cv2.drawContours(mask, contours, -1, 0, thickness=1)
    
    cv2.imwrite(f"{save_dir}binary_nuclei_map_{idx}_{number}.png", mask*255)

def process_single_patch(args):
    """
    Worker function to process a single patch file.
    This function is designed to be called in parallel.
    
    Args:
        args: Tuple of (npy_path, hex_size, idx, number, save_dir)
    
    Returns:
        Tuple of (npy_path, success, error_message)
    """
    npy_path, hex_size, idx, number, save_dir = args
    try:
        cells, locations, cells_json = convert_patch_to_hexagonal(npy_path, hex_size, idx, number, save_dir)
        save_binary_nuclei_map(npy_path, save_dir, idx, number, with_contour=True)
        return (npy_path, True, None)
    except Exception as e:
        return (npy_path, False, str(e))


def main():
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
    npy_files = npy_files[:]
    print(f"Found {len(npy_files)} .npy files to process")

    hex_size = 20
    save_dir = f"dataset/training_data/consep/consep/train/540x540_164x164/mask_original/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = []
    for npy_path in npy_files[:]:
        filename = os.path.basename(npy_path)
        match = re.search(r'train_(\d+)_(\d+)\.npy', filename)
        if match:
            idx = int(match.group(1))
            number = int(match.group(2))
            process_args.append((npy_path, hex_size, idx, number, save_dir))
    
    # Process files in parallel
    max_workers = int(os.cpu_count()/2) or 4  # Use all available CPU cores
    print(f"Processing {len(process_args)} files in parallel using {max_workers} workers...")
    
    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_patch, args): args[0] for args in process_args}
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_path):
            npy_path, success, error = future.result()
            if success:
                completed += 1
                print(f"[{completed}/{len(process_args)}] Completed: {os.path.basename(npy_path)}")
            else:
                failed += 1
                print(f"[ERROR] Failed to process {os.path.basename(npy_path)}: {error}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {completed}/{len(process_args)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(process_args)}")

def visualize_arcade_simulation(locations_data, hex_size, center_x, center_y, target_size, max_cells, output_path=None, use_white_background=True):
    """
    Visualize the arcade simulation.
    """
    plot_hexagonal_population(locations_data, hex_size, center_x, center_y, target_size, max_cells, output_path=output_path, use_white_background=use_white_background)

# Example usage
if __name__ == "__main__":
    #main()
    hex_size = 30 / 3 ** 0.5 # microns
    print(f"Hex size: {hex_size} microns")
    target_size = 540 # pixels
    center_x = target_size/2
    center_y = target_size/2
    
    use_white_background = False
    for input_id in range(1, 2):
        locations_data_path = f"../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/iter_0/inputs/input_{input_id}/combined_grid_0009_010080.LOCATIONS.json"
        with open(locations_data_path, 'r') as f:
            locations_data = json.load(f)
        max_cells = max(len(location['ids']) for location in locations_data) if locations_data else 1
        output_path = f"ARCADE_VIZ/{input_id:06d}.png"
        
        visualize_arcade_simulation(locations_data, hex_size, center_x, center_y, target_size, max_cells, output_path, use_white_background)

