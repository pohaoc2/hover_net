import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


POP_TO_CHANNEL = {
    1: "cell_type_healthy",
    2: "cell_type_cancer",
    3: "cell_type_immune",
}

PROLIF_STATES = {"PROLIFERATIVE"}
DEAD_STATES = {"APOPTOTIC", "DEAD", "NECROTIC"}

DEFAULT_HEX_SIZE_UM = 30.0 / math.sqrt(3.0)


def hex_to_xy(u, v, w, hex_size_px, center_x, center_y):
    q = u
    r = v
    x_rel = hex_size_px * (3.0 / 2.0 * q)
    y_rel = hex_size_px * (math.sqrt(3.0) / 2.0 * q + math.sqrt(3.0) * r)
    return x_rel + center_x, y_rel + center_y


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stem_from_path(path):
    name = Path(path).name
    for suffix in [".CELLS.json", ".LOCATIONS.json", ".LAYERS.json", ".json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def parse_size(value):
    if value is None:
        return None
    if "x" in value.lower():
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    size = int(value)
    return size, size


def unit_hex_xy(coords):
    points = []
    for u, v, w in coords:
        points.append((1.5 * u, math.sqrt(3.0) / 2.0 * u + math.sqrt(3.0) * v))
    return points


def build_canvas(coords, hex_size_px=None, image_size=None, padding_px=8):
    if not coords:
        raise ValueError("No hex coordinates were provided.")

    unit_points = unit_hex_xy(coords)
    min_x = min(x for x, _ in unit_points) - 1.0
    max_x = max(x for x, _ in unit_points) + 1.0
    min_y = min(y for _, y in unit_points) - 1.0
    max_y = max(y for _, y in unit_points) + 1.0
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    if image_size is not None and hex_size_px is None:
        width, height = image_size
        usable_width = max(width - 2 * padding_px, 1)
        usable_height = max(height - 2 * padding_px, 1)
        hex_size_px = min(usable_width / span_x, usable_height / span_y)
    elif hex_size_px is None:
        raise ValueError("Either hex_size_px or image_size must be provided.")

    if image_size is None:
        width = int(math.ceil(span_x * hex_size_px + 2 * padding_px))
        height = int(math.ceil(span_y * hex_size_px + 2 * padding_px))
    else:
        width, height = image_size

    center_x = (width - (min_x + max_x) * hex_size_px) / 2.0
    center_y = (height - (min_y + max_y) * hex_size_px) / 2.0

    return {
        "width": int(width),
        "height": int(height),
        "hex_size_px": float(hex_size_px),
        "center_x": float(center_x),
        "center_y": float(center_y),
    }


def resolve_mpp(mpp, hex_size_um, hex_size_px):
    if mpp is not None:
        return float(mpp)
    return float(hex_size_um) / float(hex_size_px)


def create_location_mappings(locations_data):
    id_to_location = {}
    coord_to_ids = {}
    coords = []
    for location in locations_data:
        coord = tuple(location["coordinate"])
        coord_to_ids[coord] = list(location["ids"])
        coords.append(tuple(coord[:3]))
        for cell_id in location["ids"]:
            id_to_location[cell_id] = coord
    return id_to_location, coord_to_ids, coords


def create_layer_coordinates(layers_data):
    return [tuple(entry["location"][:3]) for entry in layers_data]


def calculate_ellipse_axes(volume, height, circularity_min, circularity_max, mpp, rng):
    height = max(float(height), 1e-6)
    circularity = rng.uniform(circularity_min, circularity_max)
    area_um2 = float(volume) / height
    a_um = math.sqrt(area_um2 / (math.pi * circularity))
    b_um = circularity * a_um
    return max(1.0, a_um / mpp), max(1.0, b_um / mpp)


def get_cell_position(cell_id, hex_coord, coord_to_ids, canvas, overlap_offset_factor):
    u, v, w, _ = hex_coord
    base_x, base_y = hex_to_xy(
        u,
        v,
        w,
        canvas["hex_size_px"],
        canvas["center_x"],
        canvas["center_y"],
    )
    cell_ids = coord_to_ids.get(tuple(hex_coord), [cell_id])
    if len(cell_ids) <= 1:
        return base_x, base_y

    cell_index = cell_ids.index(cell_id)
    if cell_index == 0:
        return base_x, base_y

    directions = [0, 60, 120, 180, 240, 300]
    angle_deg = directions[(cell_index - 1) % len(directions)]
    angle_rad = math.radians(angle_deg)
    offset_distance = canvas["hex_size_px"] * overlap_offset_factor * 2.0
    offset_x = offset_distance * math.cos(angle_rad)
    offset_y = -offset_distance * math.sin(angle_rad)
    return base_x + offset_x, base_y + offset_y


def state_to_channel(state):
    state_name = str(state).upper()
    if state_name in PROLIF_STATES:
        return "cell_state_prolif"
    if state_name in DEAD_STATES:
        return "cell_state_dead"
    return "cell_state_nonprolif"


def init_binary_masks(height, width):
    masks = {
        "cell_mask": np.zeros((height, width), dtype=np.uint8),
        "cell_type_healthy": np.zeros((height, width), dtype=np.uint8),
        "cell_type_cancer": np.zeros((height, width), dtype=np.uint8),
        "cell_type_immune": np.zeros((height, width), dtype=np.uint8),
        "cell_state_prolif": np.zeros((height, width), dtype=np.uint8),
        "cell_state_nonprolif": np.zeros((height, width), dtype=np.uint8),
        "cell_state_dead": np.zeros((height, width), dtype=np.uint8),
    }
    return masks


def render_cell_masks(
    cells_data,
    locations_data,
    canvas,
    mpp,
    circularity_min,
    circularity_max,
    overlap_offset_factor,
    volume_min,
    seed,
):
    height = canvas["height"]
    width = canvas["width"]
    masks = init_binary_masks(height, width)
    id_to_location, coord_to_ids, _ = create_location_mappings(locations_data)
    rng = np.random.default_rng(seed)

    kept_cells = 0
    skipped_missing_location = 0
    skipped_small_volume = 0

    for cell in cells_data:
        cell_id = cell["id"]
        if cell_id not in id_to_location:
            skipped_missing_location += 1
            continue

        volume = float(cell.get("volume", 0.0))
        if volume < volume_min:
            skipped_small_volume += 1
            continue

        hex_coord = id_to_location[cell_id]
        x, y = get_cell_position(
            cell_id,
            hex_coord,
            coord_to_ids,
            canvas,
            overlap_offset_factor,
        )
        a, b = calculate_ellipse_axes(
            volume=volume,
            height=cell.get("height", 8.7),
            circularity_min=circularity_min,
            circularity_max=circularity_max,
            mpp=mpp,
            rng=rng,
        )
        center = (int(round(x)), int(round(y)))
        axes = (max(1, int(round(a))), max(1, int(round(b))))
        angle = float(rng.uniform(0.0, 360.0))

        type_channel = POP_TO_CHANNEL.get(int(cell.get("pop", 0)))
        state_channel = state_to_channel(cell.get("state", ""))
        targets = [masks["cell_mask"], masks[state_channel]]
        if type_channel is not None:
            targets.append(masks[type_channel])

        for target in targets:
            cv2.ellipse(target, center, axes, angle, 0, 360, 255, -1)

        kept_cells += 1

    stats = {
        "rendered_cells": kept_cells,
        "skipped_missing_location": skipped_missing_location,
        "skipped_small_volume": skipped_small_volume,
    }
    return masks, stats


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_binary_mask(mask, output_dir, basename):
    ensure_dir(output_dir)
    output_path = Path(output_dir) / f"{basename}.png"
    cv2.imwrite(str(output_path), mask)
    return str(output_path)


def normalize_float_map(values, domain_mask):
    preview = np.zeros(values.shape, dtype=np.uint8)
    inside = values[domain_mask > 0]
    if inside.size == 0:
        return preview

    min_value = float(np.min(inside))
    max_value = float(np.max(inside))
    if math.isclose(min_value, max_value):
        preview[domain_mask > 0] = 255 if max_value > 0 else 0
        return preview

    normalized = (values - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    preview = (normalized * 255.0).astype(np.uint8)
    preview[domain_mask == 0] = 0
    return preview


def save_float_map(values, domain_mask, output_dir, basename):
    ensure_dir(output_dir)
    np.save(Path(output_dir) / f"{basename}.npy", values.astype(np.float32))
    preview = normalize_float_map(values, domain_mask)
    cv2.imwrite(str(Path(output_dir) / f"{basename}.png"), preview)


def hexagon_points(u, v, w, canvas):
    center_x, center_y = hex_to_xy(
        u,
        v,
        w,
        canvas["hex_size_px"],
        canvas["center_x"],
        canvas["center_y"],
    )
    angles = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    xs = center_x + canvas["hex_size_px"] * np.cos(angles)
    ys = center_y + canvas["hex_size_px"] * np.sin(angles)
    points = np.round(np.column_stack([xs, ys])).astype(np.int32)
    return points.reshape((-1, 1, 2))


def rasterize_layer_field(layers_data, field_name, canvas):
    height = canvas["height"]
    width = canvas["width"]
    values = np.zeros((height, width), dtype=np.float32)
    domain_mask = np.zeros((height, width), dtype=np.uint8)

    for entry in layers_data:
        u, v, w, _ = entry["location"]
        points = hexagon_points(u, v, w, canvas)
        cv2.fillPoly(domain_mask, [points], 1)
        if field_name in entry["layers"]:
            cv2.fillPoly(values, [points], float(entry["layers"][field_name]))

    return values, domain_mask


def blur_and_normalize(mask, sigma_px):
    mask_float = mask.astype(np.float32) / 255.0
    if sigma_px > 0:
        mask_float = ndimage.gaussian_filter(mask_float, sigma=sigma_px)
    nonzero = mask_float[mask_float > 0]
    if nonzero.size == 0:
        return np.zeros_like(mask_float)
    max_value = float(np.max(nonzero))
    if max_value <= 0:
        return np.zeros_like(mask_float)
    return np.clip(mask_float / max_value, 0.0, 1.0)


def boundary_mask_from_domain(domain_mask):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    eroded = ndimage.binary_erosion(domain_mask.astype(bool), structure=structure)
    boundary = domain_mask.astype(bool) & ~eroded
    interior = eroded
    return boundary, interior


def solve_boundary_pde(
    domain_mask,
    prolif_mask,
    immune_mask,
    effective_mpp,
    krogh_um,
    k_base,
    prolif_weight,
    immune_weight,
    demand_sigma_px,
    max_iters,
    tol,
):
    if not np.any(domain_mask):
        raise ValueError("Domain mask is empty; cannot solve PDE.")

    domain_bool = domain_mask.astype(bool)
    boundary, interior = boundary_mask_from_domain(domain_mask)
    if not np.any(interior):
        solution = np.zeros(domain_mask.shape, dtype=np.float32)
        solution[domain_bool] = 1.0
        return solution, {"iterations": 0, "max_delta": 0.0}

    prolif_norm = blur_and_normalize(prolif_mask, demand_sigma_px)
    immune_norm = blur_and_normalize(immune_mask, demand_sigma_px)
    k_map = k_base + prolif_weight * prolif_norm + immune_weight * immune_norm
    alpha = k_map / max(k_base, 1e-6)
    decay_length_px = krogh_um / max(effective_mpp, 1e-6)
    coeff = alpha / max(decay_length_px ** 2, 1e-6)

    solution = np.zeros(domain_mask.shape, dtype=np.float32)
    solution[domain_bool] = 1.0
    max_delta = None

    for iteration in range(1, max_iters + 1):
        previous = solution.copy()
        neighbor_sum = np.zeros_like(previous)
        neighbor_sum[1:-1, 1:-1] = (
            previous[:-2, 1:-1]
            + previous[2:, 1:-1]
            + previous[1:-1, :-2]
            + previous[1:-1, 2:]
        )
        solution[interior] = neighbor_sum[interior] / (4.0 + coeff[interior])
        solution[boundary] = 1.0
        solution[~domain_bool] = 0.0
        max_delta = float(np.max(np.abs(solution[interior] - previous[interior])))
        if max_delta < tol:
            return solution, {"iterations": iteration, "max_delta": max_delta}

    return solution, {"iterations": max_iters, "max_delta": float(max_delta or 0.0)}


def layer_value_stats(values, domain_mask):
    inside = values[domain_mask > 0]
    if inside.size == 0:
        return {"min": None, "max": None, "is_constant": True}
    min_value = float(np.min(inside))
    max_value = float(np.max(inside))
    return {
        "min": min_value,
        "max": max_value,
        "is_constant": math.isclose(min_value, max_value),
    }


def save_metadata(output_dir, basename, metadata):
    ensure_dir(output_dir)
    with open(Path(output_dir) / f"{basename}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def convert_cells(args):
    cells_data = load_json(args.cells)
    locations_data = load_json(args.locations)
    _, _, coords = create_location_mappings(locations_data)

    image_size = parse_size(args.image_size) or (540, 540)
    canvas = build_canvas(
        coords=coords,
        hex_size_px=args.hex_size_px,
        image_size=image_size,
        padding_px=args.padding_px,
    )
    mpp = resolve_mpp(args.mpp, args.hex_size_um, canvas["hex_size_px"])

    masks, render_stats = render_cell_masks(
        cells_data=cells_data,
        locations_data=locations_data,
        canvas=canvas,
        mpp=mpp,
        circularity_min=args.circularity_min,
        circularity_max=args.circularity_max,
        overlap_offset_factor=args.overlap_offset_factor,
        volume_min=args.volume_min,
        seed=args.seed,
    )

    basename = args.basename or stem_from_path(args.cells)
    output_dir = Path(args.output_dir)
    saved_paths = {}
    for channel_name, mask in masks.items():
        saved_paths[channel_name] = save_binary_mask(
            mask=mask,
            output_dir=output_dir / channel_name,
            basename=basename,
        )

    metadata = {
        "mode": "cells",
        "basename": basename,
        "inputs": {
            "cells": str(Path(args.cells).resolve()),
            "locations": str(Path(args.locations).resolve()),
        },
        "canvas": canvas,
        "resolved_mpp": mpp,
        "hex_size_um": args.hex_size_um,
        "parameters": {
            "circularity_min": args.circularity_min,
            "circularity_max": args.circularity_max,
            "overlap_offset_factor": args.overlap_offset_factor,
            "volume_min": args.volume_min,
            "seed": args.seed,
        },
        "render_stats": render_stats,
        "saved_paths": saved_paths,
    }
    save_metadata(output_dir / "metadata", basename, metadata)
    return metadata


def convert_layers(args):
    if bool(args.cells) != bool(args.locations):
        raise ValueError("Provide both --cells and --locations together for demand-aware PDE mode.")

    layers_data = load_json(args.layers)
    layer_coords = create_layer_coordinates(layers_data)
    image_size = parse_size(args.image_size)
    canvas = build_canvas(
        coords=layer_coords,
        hex_size_px=args.hex_size_px,
        image_size=image_size,
        padding_px=args.padding_px,
    )
    effective_mpp = resolve_mpp(args.mpp, args.hex_size_um, canvas["hex_size_px"])
    basename = args.basename or stem_from_path(args.layers)
    output_dir = Path(args.output_dir)

    oxygen_raw, domain_mask = rasterize_layer_field(layers_data, "OXYGEN", canvas)
    glucose_raw, _ = rasterize_layer_field(layers_data, "GLUCOSE", canvas)
    oxygen_stats = layer_value_stats(oxygen_raw, domain_mask)
    glucose_stats = layer_value_stats(glucose_raw, domain_mask)

    raw_saved = {}
    if args.save_raw_layers:
        save_float_map(oxygen_raw, domain_mask, output_dir / "oxygen_raw", basename)
        save_float_map(glucose_raw, domain_mask, output_dir / "glucose_raw", basename)
        raw_saved = {
            "oxygen_raw": str(output_dir / "oxygen_raw" / f"{basename}.npy"),
            "glucose_raw": str(output_dir / "glucose_raw" / f"{basename}.npy"),
        }

    cell_metadata = None
    if args.cells and args.locations:
        cells_data = load_json(args.cells)
        locations_data = load_json(args.locations)
        cell_masks, render_stats = render_cell_masks(
            cells_data=cells_data,
            locations_data=locations_data,
            canvas=canvas,
            mpp=effective_mpp,
            circularity_min=args.circularity_min,
            circularity_max=args.circularity_max,
            overlap_offset_factor=args.overlap_offset_factor,
            volume_min=args.volume_min,
            seed=args.seed,
        )
        cell_metadata = {"render_stats": render_stats}
    else:
        cell_masks = init_binary_masks(canvas["height"], canvas["width"])

    use_pde = args.mode == "pde"
    if args.mode == "auto":
        use_pde = oxygen_stats["is_constant"] or glucose_stats["is_constant"]

    pde_metadata = {}
    if use_pde or args.mode == "both":
        demand_sigma_px = (
            args.demand_sigma_px
            if args.demand_sigma_px is not None
            else max(1.0, 0.75 * canvas["hex_size_px"])
        )
        oxygen_pde, oxygen_solver = solve_boundary_pde(
            domain_mask=domain_mask,
            prolif_mask=cell_masks["cell_state_prolif"],
            immune_mask=cell_masks["cell_type_immune"],
            effective_mpp=effective_mpp,
            krogh_um=args.oxygen_pde_krogh_um,
            k_base=args.oxygen_k_base,
            prolif_weight=args.oxygen_prolif_weight,
            immune_weight=args.oxygen_immune_weight,
            demand_sigma_px=demand_sigma_px,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        glucose_pde, glucose_solver = solve_boundary_pde(
            domain_mask=domain_mask,
            prolif_mask=cell_masks["cell_state_prolif"],
            immune_mask=cell_masks["cell_type_immune"],
            effective_mpp=effective_mpp,
            krogh_um=args.glucose_pde_krogh_um,
            k_base=args.glucose_k_base,
            prolif_weight=args.glucose_prolif_weight,
            immune_weight=args.glucose_immune_weight,
            demand_sigma_px=demand_sigma_px,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        pde_metadata = {
            "assumption": "Boundary-supplied steady-state diffusion-consumption proxy because no vasculature mask is available.",
            "effective_mpp": effective_mpp,
            "demand_sigma_px": demand_sigma_px,
            "oxygen": {
                "krogh_um": args.oxygen_pde_krogh_um,
                "k_base": args.oxygen_k_base,
                "prolif_weight": args.oxygen_prolif_weight,
                "immune_weight": args.oxygen_immune_weight,
                "solver": oxygen_solver,
            },
            "glucose": {
                "krogh_um": args.glucose_pde_krogh_um,
                "k_base": args.glucose_k_base,
                "prolif_weight": args.glucose_prolif_weight,
                "immune_weight": args.glucose_immune_weight,
                "solver": glucose_solver,
            },
        }
    else:
        oxygen_pde = None
        glucose_pde = None

    if args.mode == "raw":
        oxygen_final = oxygen_raw
        glucose_final = glucose_raw
        final_mode = "raw_layers"
    elif args.mode == "auto":
        if use_pde:
            oxygen_final = oxygen_pde
            glucose_final = glucose_pde
            final_mode = "boundary_pde_proxy"
        else:
            oxygen_final = oxygen_raw
            glucose_final = glucose_raw
            final_mode = "raw_layers"
    elif args.mode == "pde":
        oxygen_final = oxygen_pde
        glucose_final = glucose_pde
        final_mode = "boundary_pde_proxy"
    else:
        oxygen_final = oxygen_pde
        glucose_final = glucose_pde
        final_mode = "boundary_pde_proxy"
        save_float_map(oxygen_raw, domain_mask, output_dir / "oxygen_raw", basename)
        save_float_map(glucose_raw, domain_mask, output_dir / "glucose_raw", basename)
        raw_saved = {
            "oxygen_raw": str(output_dir / "oxygen_raw" / f"{basename}.npy"),
            "glucose_raw": str(output_dir / "glucose_raw" / f"{basename}.npy"),
        }

    save_float_map(oxygen_final, domain_mask, output_dir / "oxygen", basename)
    save_float_map(glucose_final, domain_mask, output_dir / "glucose", basename)

    metadata = {
        "mode": "layers",
        "basename": basename,
        "inputs": {
            "layers": str(Path(args.layers).resolve()),
            "cells": str(Path(args.cells).resolve()) if args.cells else None,
            "locations": str(Path(args.locations).resolve()) if args.locations else None,
        },
        "canvas": canvas,
        "resolved_mpp": effective_mpp,
        "hex_size_um": args.hex_size_um,
        "layer_stats": {
            "oxygen_raw": oxygen_stats,
            "glucose_raw": glucose_stats,
        },
        "selected_output_mode": final_mode,
        "pde_metadata": pde_metadata,
        "cell_metadata": cell_metadata,
        "saved_paths": {
            "oxygen": str(output_dir / "oxygen" / f"{basename}.npy"),
            "glucose": str(output_dir / "glucose" / f"{basename}.npy"),
            **raw_saved,
        },
    }
    save_metadata(output_dir / "metadata", basename, metadata)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert ARCADE outputs into mask channels and nutrient maps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cells_parser = subparsers.add_parser(
        "cells",
        help="Convert CELLS.json + LOCATIONS.json into binary cell/type/state masks.",
    )
    add_shared_cell_arguments(cells_parser)
    cells_parser.add_argument("--cells", required=True, help="Path to CELLS.json")
    cells_parser.add_argument("--locations", required=True, help="Path to LOCATIONS.json")
    cells_parser.add_argument("--output-dir", required=True, help="Directory for output channels")
    cells_parser.add_argument(
        "--image-size",
        default="540",
        help="Canvas size as N or WxH. Default: 540",
    )
    cells_parser.add_argument(
        "--hex-size-px",
        type=float,
        default=20.0,
        help="Rendered hex size in pixels. Default: 20",
    )
    cells_parser.add_argument(
        "--padding-px",
        type=int,
        default=8,
        help="Canvas padding in pixels. Default: 8",
    )
    cells_parser.set_defaults(func=convert_cells)

    layers_parser = subparsers.add_parser(
        "layers",
        help="Convert LAYERS.json into oxygen/glucose maps, optionally using a PDE proxy.",
    )
    add_shared_cell_arguments(layers_parser)
    layers_parser.add_argument("--layers", required=True, help="Path to LAYERS.json")
    layers_parser.add_argument("--cells", help="Optional CELLS.json for demand-aware PDE")
    layers_parser.add_argument("--locations", help="Optional LOCATIONS.json for demand-aware PDE")
    layers_parser.add_argument("--output-dir", required=True, help="Directory for output maps")
    layers_parser.add_argument(
        "--image-size",
        default="540",
        help="Canvas size as N or WxH. If hex-size-px is omitted, the tool fits hexes to this canvas. Default: 540",
    )
    layers_parser.add_argument(
        "--hex-size-px",
        type=float,
        default=None,
        help="Rendered hex size in pixels. If omitted, hexes are fit to the requested canvas.",
    )
    layers_parser.add_argument(
        "--padding-px",
        type=int,
        default=8,
        help="Canvas padding in pixels. Default: 8",
    )
    layers_parser.add_argument(
        "--mode",
        choices=["auto", "raw", "pde", "both"],
        default="auto",
        help="How to produce oxygen/glucose outputs. Default: auto",
    )
    layers_parser.add_argument(
        "--save-raw-layers",
        action="store_true",
        help="Also save rasterized raw OXYGEN/GLUCOSE maps when LAYERS.json contains them.",
    )
    layers_parser.add_argument(
        "--demand-sigma-px",
        type=float,
        default=None,
        help="Gaussian smoothing sigma for proliferative/immune demand maps in PDE mode.",
    )
    layers_parser.add_argument(
        "--oxygen-pde-krogh-um",
        type=float,
        default=200.0,
        help="Stage-4 oxygen decay length in microns. Default: 200",
    )
    layers_parser.add_argument(
        "--glucose-pde-krogh-um",
        type=float,
        default=120.0,
        help="Stage-4 glucose decay length in microns. Default: 120",
    )
    layers_parser.add_argument(
        "--oxygen-k-base",
        type=float,
        default=0.1,
        help="Baseline oxygen consumption coefficient. Default: 0.1",
    )
    layers_parser.add_argument(
        "--glucose-k-base",
        type=float,
        default=0.1,
        help="Baseline glucose consumption coefficient. Default: 0.1",
    )
    layers_parser.add_argument(
        "--oxygen-prolif-weight",
        type=float,
        default=0.3,
        help="Stage-4-style proliferative demand weight for oxygen. Default: 0.3",
    )
    layers_parser.add_argument(
        "--glucose-prolif-weight",
        type=float,
        default=0.3,
        help="Stage-4-style proliferative demand weight for glucose. Default: 0.3",
    )
    layers_parser.add_argument(
        "--oxygen-immune-weight",
        type=float,
        default=0.1,
        help="Stage-4-style immune demand weight for oxygen. Default: 0.1",
    )
    layers_parser.add_argument(
        "--glucose-immune-weight",
        type=float,
        default=0.1,
        help="Stage-4-style immune demand weight for glucose. Default: 0.1",
    )
    layers_parser.add_argument(
        "--max-iters",
        type=int,
        default=2000,
        help="Maximum Jacobi iterations for PDE mode. Default: 2000",
    )
    layers_parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Jacobi convergence tolerance for PDE mode. Default: 1e-4",
    )
    layers_parser.set_defaults(func=convert_layers)

    return parser


def add_shared_cell_arguments(parser):
    parser.add_argument(
        "--basename",
        default=None,
        help="Override output basename. Default: derived from the input filename",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help="Microns per pixel. If omitted, derived from hex-size-um / hex-size-px.",
    )
    parser.add_argument(
        "--hex-size-um",
        type=float,
        default=DEFAULT_HEX_SIZE_UM,
        help=f"Physical hex radius in microns. Default: {DEFAULT_HEX_SIZE_UM:.6f}",
    )
    parser.add_argument(
        "--circularity-min",
        type=float,
        default=1.0,
        help="Minimum ellipse circularity. Default: 1.0",
    )
    parser.add_argument(
        "--circularity-max",
        type=float,
        default=1.0,
        help="Maximum ellipse circularity. Default: 1.0",
    )
    parser.add_argument(
        "--overlap-offset-factor",
        type=float,
        default=0.8,
        help="Relative offset for multiple cells occupying the same hex. Default: 0.8",
    )
    parser.add_argument(
        "--volume-min",
        type=float,
        default=1.0,
        help="Skip cells smaller than this volume in um^3. Default: 1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for ellipse orientation and circularity. Default: 42",
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    metadata = args.func(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
