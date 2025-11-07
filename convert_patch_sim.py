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

def visualize_hexagonal_conversion(img, cells, hex_size, locations_data, center, output_path=None):
    """
    Visualize the conversion from Cartesian to hexagonal coordinates overlaid on original image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    colors = get_class_colors()
    center_x, center_y = center
    
    # Plot 1: Original Cartesian coordinates overlaid on image
    ax1 = axes[0]
    ax1.imshow(img)
    for cell in cells:
        color = np.array(colors.get(cell['type'], (128, 128, 128))) / 255.0
        ax1.scatter(cell['x'], cell['y'], c=[color], s=50, alpha=0.8, edgecolors='k', linewidths=1)
    
    
    ax1.set_xlabel('X (pixels)', fontsize=12)
    ax1.set_ylabel('Y (pixels)', fontsize=12)
    ax1.set_title('Original Cartesian Coordinates', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Plot 2: Hexagonal grid with assigned cells overlaid on image
    ax2 = axes[1]
    ax2.imshow(img)
    
    # Draw hexagonal grid
    for location in locations_data:
        u, v, w, _ = location['coordinate']
        x, y = hex_to_xy(u, v, w, hex_size, center_x, center_y)
        
        # Draw hexagon
        angles = np.linspace(0, 2*np.pi, 7)
        hex_x = x + hex_size * np.cos(angles)
        hex_y = y + hex_size * np.sin(angles)
        ax2.plot(hex_x, hex_y, 'k', alpha=0.6, linewidth=2)
        
        # Plot cells in this hexagon
        num_cells_in_hex = len(location['ids'])
        # cell: list of dicts with id, x, y, type, area
        cell_types = [c['type'] for c in cells if c['id'] in location['ids']] 
        most_common_type = max(set(cell_types), key=cell_types.count)
        color = np.array(colors.get(most_common_type, (128, 128, 128))) / 255.0
        ax2.scatter(x, y, c=[color], s=50, alpha=0.8, edgecolors='k', linewidths=1)

    
    ax2.set_xlabel('X (pixels)', fontsize=12)
    ax2.set_ylabel('Y (pixels)', fontsize=12)
    ax2.set_title(f'Hexagonal Grid (hex_size={hex_size}px)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add legend for cell types
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
    fig.legend(handles=legend_elements, loc='upper center', ncol=7, 
              bbox_to_anchor=(0.5, 0.98), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    plt.close()

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
    print(f"Extracted {len(cells)} cells")
    
    # Convert to hexagonal system
    locations_data, cells_data, center = convert_to_hexagonal_system(
        cells, img.shape, hex_size, idx, number, save_dir   
    )
    
    # Visualize
    visualize_hexagonal_conversion(img, cells, hex_size, locations_data, center,
                                   output_path=f"{save_dir}train_{idx}_{number}_hexagonal_visualization.png")
    
    return cells, locations_data, cells_data


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
        return (npy_path, True, None)
    except Exception as e:
        return (npy_path, False, str(e))


# Example usage
if __name__ == "__main__":
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

    hex_size = 20
    save_dir = f"dataset/training_data/consep/consep/train/540x540_164x164/ARCADE_OUTPUT/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = []
    for npy_path in npy_files[49*8:]:
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
