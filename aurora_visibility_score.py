from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from branca.colormap import LinearColormap

# Data sources (hard-coded paths/URLs)
CLOUD_JSON = Path("CLOUD_OPTICAL_THICKNESS.json")
URBAN_JSON = Path("URBAN_DENSITY.json")
NO2_JSON = Path("NO2.json")
AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"

# Normalization constants (match folium_overlay.py defaults)
CLOUD_MAX = 250.0
URBAN_MAX = 1.0
NO2_MAX = 1.0e-4
AURORA_MAX = 100.0

# Output
SCORE_HEATMAP_PATH = Path("aurora_visibility_scores.png")
SCORE_MAP_PATH = Path("aurora_visibility_map.html")
SCORE_COLORS = ["#dcfce7", "#bbf7d0", "#4ade80", "#22c55e", "#166534"]


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


def load_aurora_overlay(url: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
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
        intermediate[i] = np.interp(
            target_lons,
            source_lons,
            row,
            left=np.nan,
            right=np.nan,
        )

    resampled_asc = np.empty((target_lats_asc.size, target_lons.size), dtype=np.float32)
    for j in range(target_lons.size):
        column = intermediate[:, j]
        resampled_asc[:, j] = np.interp(
            target_lats_asc,
            source_lats,
            column,
            left=np.nan,
            right=np.nan,
        )

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


def create_folium_map(score_grid: np.ndarray, bbox: list[float], output_path: Path) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    rgba = grid_to_rgba(score_grid, SCORE_COLORS)
    image_url = rgba_to_data_url(rgba)

    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    folium.raster_layers.ImageOverlay(
        name="Aurora visibility score",
        image=image_url,
        bounds=bounds,
        opacity=0.85,
    ).add_to(fmap)

    colormap = LinearColormap(colors=SCORE_COLORS, vmin=0.0, vmax=1.0)
    colormap.caption = "Aurora visibility score"
    colormap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.save(output_path)
    print(f"Saved Folium map to {output_path.resolve()}")


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


def main() -> None:
    cloud_payload, cloud_grid = load_grid(CLOUD_JSON, "cloud_optical_thickness")
    urban_payload, urban_grid = load_grid(URBAN_JSON, "urban_density")
    no2_payload, no2_grid = load_grid(NO2_JSON, "no2")
    bbox_ref = cloud_payload["bbox"]
    if bbox_ref != urban_payload["bbox"] or bbox_ref != no2_payload["bbox"]:
        raise ValueError("Cloud, urban, and NO2 datasets must share the same bbox.")

    lats, lons, aurora_grid = load_aurora_overlay(AURORA_URL)
    width = cloud_payload.get("width", cloud_grid.shape[1])
    height = cloud_payload.get("height", cloud_grid.shape[0])
    target_lons, target_lats_desc = compute_target_coords(bbox_ref, width, height)

    aurora_resampled = resample_aurora(lats, lons, aurora_grid, target_lats_desc, target_lons)
    scores = compute_scores(cloud_grid, urban_grid, aurora_resampled, no2_grid=no2_grid)

    valid_scores = scores[np.isfinite(scores)]
    if valid_scores.size:
        print(
            f"Score stats - min: {valid_scores.min():.3f}, "
            f"mean: {valid_scores.mean():.3f}, max: {valid_scores.max():.3f}"
        )
    else:
        print("No valid scores computed.")

    plot_heatmap(scores, bbox_ref, SCORE_HEATMAP_PATH)
    create_folium_map(scores, bbox_ref, SCORE_MAP_PATH)


if __name__ == "__main__":
    main()
