from __future__ import annotations

import argparse
import base64
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import folium
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from branca.colormap import LinearColormap

# --------------------------------------------------------------------------------------
# Data sources (hard-coded paths/URLs)
# --------------------------------------------------------------------------------------
CLOUD_JSON = Path("CLOUD_OPTICAL_THICKNESS.json")
URBAN_JSON = Path("URBAN_DENSITY.json")
NO2_JSON = Path("NO2.json")
AURORA_JSON = Path("AURORA_SIMULATED.json")
AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

# Normalization constants
CLOUD_MAX = 250.0
URBAN_MAX = 1.0
NO2_MAX = 1.0e-4
AURORA_MAX = 100.0

# Color ramps (match folium_overlay.py defaults)
CLOUD_COLORS = ["#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#bb3e03"]
URBAN_COLORS = ["#001233", "#004b8d", "#3465a4", "#4f772d", "#b7e4c7"]
NO2_COLORS = [
    "#e0f7fa",  # light cyan
    "#b2ebf2",  # pale blue
    "#4dd0e1",  # medium turquoise
    "#0288d1",  # bright blue
    "#01579b",  # deep ocean blue
]
SCORE_COLORS = ["#dcfce7", "#bbf7d0", "#4ade80", "#22c55e", "#166534"]
AURORA_COLORS = SCORE_COLORS

# Output
SCORE_HEATMAP_PATH = Path("aurora_visibility_scores.png")
SCORE_MAP_PATH = Path("aurora_visibility_map.html")

# Selected points export
SELECTED_POINTS_JSON = Path("AURORA_VISIBILITY_SELECTED.json")
SELECTED_PERCENTILE = 99.5
SELECTED_MAX_POINTS = 500

# Optional VIIRS Black Marble night-lights overlay
BLACK_MARBLE_TIF = Path("BlackMarble_2016_3km_geo.tif")
BLACK_MARBLE_NAME = "Black Marble night lights"
BLACK_MARBLE_COLORS = [
    "#02020a",  # space black with blue tint
    "#061639",  # deep midnight blue
    "#1c1e5c",  # indigo shadows
    "#37306e",  # muted violet landmass
    "#735f3d",  # dim urban glow
    "#f8edb1",  # bright city core
]
BLACK_MARBLE_PERCENTILE_MIN = 2.0
BLACK_MARBLE_PERCENTILE_MAX = 99.0

# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def load_grid(path: Path, band_key: str) -> tuple[dict, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if band_key not in payload:
        raise KeyError(f"Band '{band_key}' missing in {path.name}.")
    grid = np.array(payload[band_key], dtype=np.float32)
    mask = payload.get("data_mask")
    if mask is not None:
        mask_arr = np.array(mask, dtype=np.float32)
        grid = np.where(mask_arr >= 0.5, grid, np.nan)
    return payload, grid


def _build_aurora_grid(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = payload.get("coordinates")
    if not coords:
        raise ValueError("Aurora response missing 'coordinates'.")
    arr = np.asarray(coords, dtype=np.float32)
    lons = np.unique(arr[:, 0])
    lats = np.unique(arr[:, 1])
    lon_to_idx = {lon: idx for idx, lon in enumerate(lons)}
    lat_to_idx = {lat: idx for idx, lat in enumerate(lats)}
    grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    for lon, lat, probability in arr:
        grid[lat_to_idx[float(lat)], lon_to_idx[float(lon)]] = float(probability)
    return lats, lons, grid


def load_aurora_overlay(use_real: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if use_real:
        response = requests.get(AURORA_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    else:
        if not AURORA_JSON.exists():
            raise FileNotFoundError(f"Aurora file '{AURORA_JSON}' not found.")
        payload = json.loads(AURORA_JSON.read_text(encoding="utf-8"))
    lats, lons, grid = _build_aurora_grid(payload)
    return lats, lons, grid, payload


def load_black_marble(path: Path) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    try:
        import rasterio
    except ImportError as exc:  # optional dependency
        raise RuntimeError(
            "Black Marble overlay requires the 'rasterio' package. Please install it first."
        ) from exc
    if not path.exists():
        raise FileNotFoundError(f"Black Marble TIFF '{path}' not found.")
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype(np.float32)
        transform = src.transform
        bounds = src.bounds
    grid = np.array(data.filled(np.nan), dtype=np.float32)
    height, width = grid.shape
    cols = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    lon_centers = transform.c + (cols + 0.5) * transform.a
    lat_centers_desc = transform.f + (rows + 0.5) * transform.e
    bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
    return bbox, lat_centers_desc, lon_centers, grid


# --------------------------------------------------------------------------------------
# Grid utilities
# --------------------------------------------------------------------------------------
def compute_target_coords(bbox: list[float], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_edges = np.linspace(min_lon, max_lon, width + 1)
    lat_edges = np.linspace(min_lat, max_lat, height + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers_desc = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    return lon_centers, lat_centers_desc[::-1]  # descending to match north-up grids


def resample_aurora(
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    source_grid: np.ndarray,
    target_lats_desc: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    target_lats_asc = target_lats_desc[::-1]
    intermediate = np.empty((source_grid.shape[0], target_lons.size), dtype=np.float32)
    for i, row in enumerate(source_grid):
        intermediate[i] = np.interp(target_lons, source_lons, row, left=np.nan, right=np.nan)
    resampled_asc = np.empty((target_lats_asc.size, target_lons.size), dtype=np.float32)
    for j in range(target_lons.size):
        column = intermediate[:, j]
        resampled_asc[:, j] = np.interp(target_lats_asc, source_lats, column, left=np.nan, right=np.nan)
    return resampled_asc[::-1]  # return north-up


def normalize_grid(grid: np.ndarray, vmax: float, invert: bool = False) -> np.ndarray:
    normalized = np.clip(grid / vmax, 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized


def compute_scores(
    cloud_grid: np.ndarray,
    urban_grid: np.ndarray,
    aurora_grid: np.ndarray,
    no2_grid: np.ndarray | None = None,
) -> np.ndarray:
    aurora_norm = normalize_grid(aurora_grid, AURORA_MAX)
    clear_sky = normalize_grid(cloud_grid, CLOUD_MAX, invert=True)
    dark_sky = normalize_grid(urban_grid, URBAN_MAX, invert=True)
    components = aurora_norm * clear_sky * dark_sky
    if no2_grid is not None:
        clean_air = normalize_grid(no2_grid, NO2_MAX, invert=True)
        components = components * clean_air
    score = components
    score[~np.isfinite(score)] = np.nan
    return score


def select_score_points(
    score_grid: np.ndarray,
    target_lats_desc: np.ndarray,
    target_lons: np.ndarray,
    percentile: float = SELECTED_PERCENTILE,
    max_points: int = SELECTED_MAX_POINTS,
) -> tuple[list[list[float]], float]:
    valid_mask = np.isfinite(score_grid)
    if not np.any(valid_mask):
        return [], float("nan")
    valid_scores = score_grid[valid_mask]
    # tighten cutoff slightly to reduce highlighted area
    adj_percentile = min(100.0, percentile + 0.4)
    threshold = float(np.percentile(valid_scores, adj_percentile)) if valid_scores.size else float("nan")
    candidate_mask = valid_mask & (score_grid >= threshold)
    candidate_indices = np.column_stack(np.where(candidate_mask))
    points = [
        [float(target_lons[c]), float(target_lats_desc[r]), round(float(score_grid[r, c]), 4)]
        for r, c in candidate_indices
    ]
    points.sort(key=lambda x: x[2], reverse=True)
    if max_points and len(points) > max_points:
        points = points[:max_points]
    return points, threshold


# --------------------------------------------------------------------------------------
# Rendering helpers (shared with folium_overlay.py features)
# --------------------------------------------------------------------------------------
def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{value}'.")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def grid_to_rgba(
    grid: np.ndarray,
    colors: list[str],
    vmin: float = 0.0,
    vmax: float = 1.0,
    max_alpha: int = 220,
) -> np.ndarray:
    if np.isclose(vmin, vmax):
        raise ValueError("vmin and vmax must differ.")
    norm = (grid - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)
    valid = np.isfinite(grid)
    if not np.any(valid):
        return rgba
    ramp_positions = np.linspace(0.0, 1.0, len(colors))
    ramp_rgb = np.array([_hex_to_rgb(c) for c in colors], dtype=np.float32)
    norm_valid = norm[valid]
    for channel in range(3):
        channel_values = np.interp(norm_valid, ramp_positions, ramp_rgb[:, channel])
        rgba[..., channel][valid] = channel_values.astype(np.uint8)
    alpha = (norm_valid * max_alpha).astype(np.uint8)
    rgba[..., 3][valid] = alpha
    return rgba


def rgba_to_data_url(rgba: np.ndarray) -> str:
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# Sentinel Hub-like highlight compression for EO-urban style
def highlight_compress(
    array: np.ndarray,
    max_input: float = 0.8,
    clip_input: float = 0.9,
    max_output: float = 1.0,
) -> np.ndarray:
    if clip_input <= max_input:
        raise ValueError("clip_input must exceed max_input.")
    arr = np.asarray(array, dtype=np.float32)
    result = np.zeros_like(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    arr = np.clip(arr, 0.0, None)
    low = valid & (arr <= max_input)
    result[low] = arr[low] / max_input
    mid = valid & (arr > max_input) & (arr < clip_input)
    result[mid] = 1.0 + ((arr[mid] - max_input) / (clip_input - max_input)) * (max_output - 1.0)
    high = valid & (arr >= clip_input)
    result[high] = max_output
    return np.clip(result, 0.0, max_output)


def build_urban_overlay_from_json(payload: dict, flip: bool) -> Optional[str]:
    """
    Recreate EO Browser Sentinel-1 styling directly from the JSON payload.
    Falls back to None if required bands are missing; caller should then render URBAN grid.
    """
    required = ("urban_density", "vv_backscatter", "vh_backscatter", "data_mask")
    if any(key not in payload for key in required):
        return None
    density = np.array(payload["urban_density"], dtype=np.float32)
    vv = np.array(payload["vv_backscatter"], dtype=np.float32)
    vh = np.array(payload["vh_backscatter"], dtype=np.float32)
    mask = np.array(payload["data_mask"], dtype=np.float32)
    is_urban = np.nan_to_num(density, nan=0.0)
    vv_channel = np.nan_to_num(vv, nan=0.0)
    vh_blue = np.nan_to_num(vh * 8.0, nan=0.0)
    composite = np.stack([is_urban, vv_channel, vh_blue], axis=-1)
    compressed = highlight_compress(composite, max_input=0.8, clip_input=0.9, max_output=1.0)
    rgba = np.zeros(composite.shape[:-1] + (4,), dtype=np.uint8)
    rgba[..., :3] = (np.clip(compressed, 0.0, 1.0) * 255).astype(np.uint8)
    valid_mask = mask > 0.5
    rgba[~valid_mask] = 0
    rgba[..., 3][valid_mask] = 200
    if flip:
        rgba = np.flipud(rgba)
    return rgba_to_data_url(rgba)


# --------------------------------------------------------------------------------------
# Folium overlay helpers
# --------------------------------------------------------------------------------------
def add_overlay(
    fmap: folium.Map,
    name: str,
    grid: Optional[np.ndarray],
    bbox: list[float],
    colors: Optional[list[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    opacity: float = 0.8,
    show_colormap: bool = True,
    image_data_url: Optional[str] = None,
) -> None:
    """
    Adds a raster overlay to the map, either from an RGBA image data URL or from a scalar grid.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    if image_data_url is not None:
        url = image_data_url
    else:
        if grid is None or colors is None or vmin is None or vmax is None:
            raise ValueError(f"Overlay '{name}' missing grid/range/colors.")
        rgba = grid_to_rgba(grid, colors, vmin=vmin, vmax=vmax, max_alpha=220)
        url = rgba_to_data_url(rgba)
    folium.raster_layers.ImageOverlay(
        name=name,
        image=url,
        bounds=bounds,
        opacity=opacity,
        interactive=False,
        cross_origin=False,
    ).add_to(fmap)
    if show_colormap and image_data_url is None:
        colormap = LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
        colormap.caption = name
        colormap.add_to(fmap)


def create_folium_map(
    score_grid: np.ndarray,
    bbox: list[float],
    output_path: Path,
    aurora_grid: np.ndarray | None = None,
    aurora_name: str | None = None,
    black_marble: dict | None = None,
    cloud_grid: np.ndarray | None = None,
    urban_grid: np.ndarray | None = None,
    no2_grid: np.ndarray | None = None,
    urban_eo_image: str | None = None,
) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    fmap = folium.Map(location=center, tiles="CartoDB dark_matter")
    fmap.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

    if cloud_grid is not None:
        add_overlay(
            fmap, "Cloud optical thickness", cloud_grid, bbox,
            colors=CLOUD_COLORS, vmin=0.0, vmax=CLOUD_MAX, opacity=0.8
        )

    if urban_eo_image is not None:
        add_overlay(
            fmap, "Urban density (EO style)", None, bbox,
            image_data_url=urban_eo_image, show_colormap=False, opacity=0.8
        )
    elif urban_grid is not None:
        add_overlay(
            fmap, "Urban density", urban_grid, bbox,
            colors=URBAN_COLORS, vmin=0.0, vmax=URBAN_MAX, opacity=0.8
        )

    if no2_grid is not None:
        add_overlay(
            fmap, "NO₂ concentration", no2_grid, bbox,
            colors=NO2_COLORS, vmin=0.0, vmax=NO2_MAX, opacity=0.8
        )

    if aurora_grid is not None and aurora_name:
        add_overlay(
            fmap, aurora_name, aurora_grid, bbox,
            colors=AURORA_COLORS, vmin=0.0, vmax=AURORA_MAX, opacity=0.75
        )

    if black_marble is not None:
        bm = black_marble
        add_overlay(
            fmap, bm["name"], bm["grid"], bbox,
            colors=BLACK_MARBLE_COLORS, vmin=bm["vmin"], vmax=bm["vmax"], opacity=0.8
        )

    add_overlay(
        fmap, "Aurora visibility score", score_grid, bbox,
        colors=SCORE_COLORS, vmin=0.0, vmax=1.0, opacity=0.85
    )

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(output_path)
    print(f"Saved full composite map to {output_path.resolve()}")


# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------
def plot_heatmap(score_grid: np.ndarray, bbox: list[float], output_path: Path) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        score_grid,
        extent=[min_lon, max_lon, min_lat, max_lat],
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, label="Aurora visibility score")
    plt.title("Aurora Visibility Score")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {output_path.resolve()}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute aurora visibility scores and build composite map.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use live NOAA aurora probabilities instead of simulated data.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Load JSON grids
    cloud_payload, cloud_grid = load_grid(CLOUD_JSON, "cloud_optical_thickness")
    urban_payload, urban_grid = load_grid(URBAN_JSON, "urban_density")
    no2_payload, no2_grid = load_grid(NO2_JSON, "no2")

    bbox_ref = cloud_payload["bbox"]
    if bbox_ref != urban_payload["bbox"] or bbox_ref != no2_payload["bbox"]:
        raise ValueError("Cloud, urban, and NO2 datasets must share the same bbox.")

    # Aurora grid (live or simulated), then resample to target grid
    lats, lons, aurora_grid, aurora_payload = load_aurora_overlay(args.real)
    width = cloud_payload.get("width", cloud_grid.shape[1])
    height = cloud_payload.get("height", cloud_grid.shape[0])
    target_lons, target_lats_desc = compute_target_coords(bbox_ref, width, height)
    aurora_resampled = resample_aurora(lats, lons, aurora_grid, target_lats_desc, target_lons)

    # Composite score
    scores = compute_scores(cloud_grid, urban_grid, aurora_resampled, no2_grid=no2_grid)
    valid_scores = scores[np.isfinite(scores)]
    if valid_scores.size:
        print(
            f"Score stats - min: {valid_scores.min():.3f}, "
            f"mean: {valid_scores.mean():.3f}, max: {valid_scores.max():.3f}"
        )
    else:
        print("No valid scores computed.")

    # Export selected top points
    selected_points, threshold = select_score_points(scores, target_lats_desc, target_lons)
    threshold_value = float(threshold) if math.isfinite(threshold) else None
    selected_output = {
        "Observation Time": aurora_payload.get("Observation Time"),
        "Forecast Time": aurora_payload.get("Forecast Time"),
        "bbox": bbox_ref,
        "score_percentile": SELECTED_PERCENTILE,
        "score_threshold": threshold_value,
        "coordinates": selected_points,
    }
    SELECTED_POINTS_JSON.write_text(json.dumps(selected_output), encoding="utf-8")
    print(f"Saved {len(selected_points)} high-score points to {SELECTED_POINTS_JSON.resolve()}")

    # Optional Black Marble overlay
    black_marble_overlay = None
    if BLACK_MARBLE_TIF.exists():
        try:
            _, bm_lats_desc, bm_lons, bm_grid = load_black_marble(BLACK_MARBLE_TIF)
        except Exception as exc:
            print(f"Failed to load Black Marble overlay: {exc}")
        else:
            bm_lats_asc = bm_lats_desc[::-1]
            bm_grid_asc = bm_grid[::-1]
            bm_resampled = resample_aurora(bm_lats_asc, bm_lons, bm_grid_asc, target_lats_desc, target_lons)
            valid = bm_resampled[np.isfinite(bm_resampled)]
            if valid.size:
                vmin = float(np.percentile(valid, BLACK_MARBLE_PERCENTILE_MIN))
                vmax = float(np.percentile(valid, BLACK_MARBLE_PERCENTILE_MAX))
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1.0
            else:
                vmin, vmax = 0.0, 1.0
            black_marble_overlay = {
                "grid": bm_resampled,
                "name": BLACK_MARBLE_NAME,
                "vmin": vmin,
                "vmax": vmax,
            }
    else:
        print(f"Black Marble TIFF '{BLACK_MARBLE_TIF}' not found; skipping night-lights overlay.")

    # Optional EO-style urban composite if extra bands exist
    urban_eo_image = build_urban_overlay_from_json(urban_payload, flip=False)

    # Plot and map
    plot_heatmap(scores, bbox_ref, SCORE_HEATMAP_PATH)
    aurora_layer_name = "NOAA aurora probability" if args.real else "NOAA aurora probability"
    create_folium_map(
        scores,
        bbox_ref,
        SCORE_MAP_PATH,
        aurora_grid=aurora_resampled,
        aurora_name=aurora_layer_name,
        black_marble=black_marble_overlay,
        cloud_grid=cloud_grid,
        urban_grid=urban_grid,
        no2_grid=no2_grid,
        urban_eo_image=urban_eo_image,
    )


if __name__ == "__main__":
    main()
