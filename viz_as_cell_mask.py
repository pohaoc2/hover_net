import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HEX_SIZE = 25
IMAGE_SIZE = 512
CIRCULARITY_MIN = 0.7
CIRCULARITY_MAX = 1.0
OVERLAP_OFFSET_FACTOR = 0.3
RANDOM_SEED = 42
FIG_WIDTH = 20
FIG_HEIGHT = 10
DPI = 150


def hex_to_xy(u, v, w, hex_size, center_x, center_y):
    """Convert hexagonal (cube) coordinates to Cartesian coordinates."""
    q = u
    r = v
    
    x_rel = hex_size * (3.0/2.0 * q)
    y_rel = hex_size * (np.sqrt(3)/2 * q + np.sqrt(3) * r)
    
    x = x_rel + center_x
    y = y_rel + center_y
    
    return x, y

def calculate_ellipse_axes(volume, height, circularity_min, circularity_max):
    """
    Calculate ellipse semi-axes (a, b) from volume and height.
    Volume = π * a * b * height
    """
    circularity = np.random.uniform(circularity_min, circularity_max)
    area = volume / height
    a = np.sqrt(area / (np.pi * circularity))
    b = circularity * a
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

def draw_cell_visualization(ax, cells_data, id_to_location, locations_data, 
                           hex_size, img_size, center_x, center_y,
                           circularity_min, circularity_max, overlap_offset_factor):
    """Draw colored cell visualization with labels."""
    for cell in cells_data:
        cell_id = cell['id']
        volume = cell['volume']
        height = cell['height']
        hex_coord = id_to_location[cell_id]
        
        # Get position
        x, y = get_cell_position(cell_id, hex_coord, id_to_location, 
                                locations_data, hex_size, center_x, center_y,
                                overlap_offset_factor)
        
        # Calculate ellipse parameters
        a, b = calculate_ellipse_axes(volume, height, circularity_min, circularity_max)
        angle = np.random.uniform(0, 360)
        
        # Draw ellipse
        ellipse = Ellipse((x, y), width=2*a, height=2*b, 
                         angle=angle, 
                         facecolor='cyan', 
                         edgecolor='blue', 
                         alpha=0.6,
                         linewidth=1.5)
        ax.add_patch(ellipse)
        
        # Add cell ID label
        ax.text(x, y, str(cell_id), 
               ha='center', va='center', 
               fontsize=8, fontweight='bold')
    
    # Set axis properties
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Cell Visualization in Hexagonal Grid')
    ax.invert_yaxis()

def draw_binary_mask(ax, cells_data, id_to_location, locations_data,
                    hex_size, img_size, center_x, center_y,
                    circularity_min, circularity_max, overlap_offset_factor):
    """Draw binary mask of cells."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    for cell in cells_data:
        cell_id = cell['id']
        volume = cell['volume']
        height = cell['height']
        hex_coord = id_to_location[cell_id]
        
        # Get position
        x, y = get_cell_position(cell_id, hex_coord, id_to_location,
                                locations_data, hex_size, center_x, center_y,
                                overlap_offset_factor)
        
        # Calculate ellipse parameters
        a, b = calculate_ellipse_axes(volume, height, circularity_min, circularity_max)
        angle = np.random.uniform(0, 360)
        
        # Rasterize ellipse
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Sample points in bounding box
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
                    mask[py, px] = 255
    
    ax.imshow(mask, cmap='gray')
    ax.set_title('Binary Mask')
    ax.axis('off')

def create_id_to_location_mapping(locations_data):
    """Create mapping from cell ID to location."""
    id_to_location = {}
    for loc in locations_data:
        coord = loc['coordinate']
        cell_ids = loc['ids']
        for cell_id in cell_ids:
            id_to_location[cell_id] = coord
    return id_to_location


def main():
    # Load data
    cell_path = '../ARCADE_OUTPUT/ABC_SMC_RF_N128_combined_grid_breast/iter_0/inputs/input_1/combined_grid_0000_000000.CELLS.json'
    location_path = '../ARCADE_OUTPUT/ABC_SMC_RF_N128_combined_grid_breast/iter_0/inputs/input_1/combined_grid_0000_000000.LOCATIONS.json'
    with open(cell_path, 'r') as f:
        cells_data = json.load(f)
    
    with open(location_path, 'r') as f:
        locations_data = json.load(f)
    
    # Derived parameters
    center_x = center_y = IMAGE_SIZE // 2
    
    # Create mapping
    id_to_location = create_id_to_location_mapping(locations_data)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Draw visualization
    draw_cell_visualization(ax1, cells_data, id_to_location, locations_data,
                           HEX_SIZE, IMAGE_SIZE, center_x, center_y,
                           CIRCULARITY_MIN, CIRCULARITY_MAX, OVERLAP_OFFSET_FACTOR)
    
    # Reset random seed for binary mask
    np.random.seed(RANDOM_SEED)
    
    # Draw binary mask
    draw_binary_mask(ax2, cells_data, id_to_location, locations_data,
                    HEX_SIZE, IMAGE_SIZE, center_x, center_y,
                    CIRCULARITY_MIN, CIRCULARITY_MAX, OVERLAP_OFFSET_FACTOR)
    
    plt.tight_layout()
    plt.savefig('../ARCADE_OUTPUT/outputs/cell_visualization.png', dpi=DPI, bbox_inches='tight')
    print("Visualization saved to cell_visualization.png")
    plt.show()

if __name__ == '__main__':
    main()