from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import folium
import numpy as np
import requests
from branca.colormap import LinearColormap
from PIL import Image

CLOUD_COLORS = ["#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#bb3e03"]
URBAN_COLORS = ["#001233", "#004b8d", "#3465a4", "#4f772d", "#b7e4c7"]
NO2_COLORS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
AURORA_COLORS = ["#dcfce7", "#bbf7d0", "#4ade80", "#22c55e", "#166534"]

CLOUD_JSON = Path("CLOUD_OPTICAL_THICKNESS.json")
CLOUD_BAND_KEY = "cloud_optical_thickness"
CLOUD_MIN = 0.0
CLOUD_MAX = 250.0

URBAN_JSON = Path("URBAN_DENSITY.json")
URBAN_BAND_KEY = "urban_density"
URBAN_MIN = 0.0
URBAN_MAX = 1.0

NO2_JSON = Path("NO2.json")
NO2_BAND_KEY = "no2"
NO2_MIN = 0.0
NO2_MAX = 1.0e-4

OUTPUT_HTML = Path("cloud_urban_overlay.html")
MAP_ZOOM = 7
GRID_ORIGIN = "north_up"

INCLUDE_AURORA = True
AURORA_JSON = Path("AURORA_SIMULATED.json")
AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
AURORA_NAME = "Simulated aurora probability"
AURORA_MIN = 0.0
AURORA_MAX = 100.0


def load_grid(path: Path, band_key: str) -> tuple[dict, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if band_key not in payload:
        raise KeyError(
            f"Band key '{band_key}' not found in {path.name}. "
            f"Available keys: {', '.join(sorted(payload.keys()))}"
        )

    grid = np.array(payload[band_key], dtype=np.float32)
    mask = payload.get("data_mask")
    if mask is not None:
        mask_arr = np.array(mask, dtype=np.float32)
        grid = np.where(mask_arr >= 0.5, grid, np.nan)
    return payload, grid


def orient_grid(grid: np.ndarray, origin: str) -> np.ndarray:
    return np.flipud(grid) if origin == "south_up" else grid


def _build_aurora_grid(payload: dict) -> tuple[dict, np.ndarray]:
    coords = payload.get("coordinates")
    if not coords:
        raise ValueError("Aurora response missing 'coordinates' entries.")

    arr = np.asarray(coords, dtype=np.float32)
    lons = np.unique(arr[:, 0])
    lats = np.unique(arr[:, 1])
    lon_to_idx = {lon: idx for idx, lon in enumerate(lons)}
    lat_to_idx = {lat: idx for idx, lat in enumerate(lats)}

    grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    for lon, lat, probability in coords:
        grid[lat_to_idx[float(lat)], lon_to_idx[float(lon)]] = float(probability)

    bbox = [float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())]
    aurora_payload = {
        "bbox": payload.get("bbox", bbox),
        "observation_time": payload.get("Observation Time"),
        "forecast_time": payload.get("Forecast Time"),
    }
    return aurora_payload, grid


def load_aurora_overlay(use_real: bool) -> tuple[dict, np.ndarray]:
    if use_real:
        response = requests.get(AURORA_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    else:
        if not AURORA_JSON.exists():
            raise FileNotFoundError(f"Aurora file '{AURORA_JSON}' not found.")
        payload = json.loads(AURORA_JSON.read_text(encoding="utf-8"))
    return _build_aurora_grid(payload)


def highlight_compress(
    array: np.ndarray,
    max_input: float = 0.8,
    clip_input: float = 0.9,
    max_output: float = 1.0,
) -> np.ndarray:
    """Replicate Sentinel Hub highlight compression for positive rasters."""
    if clip_input <= max_input:
        raise ValueError("clip_input must exceed max_input.")

    arr = np.asarray(array, dtype=np.float32)
    result = np.zeros_like(arr, dtype=np.float32)

    valid = np.isfinite(arr)
    arr = np.clip(arr, 0.0, None)

    low = valid & (arr <= max_input)
    result[low] = arr[low] / max_input

    mid = valid & (arr > max_input) & (arr < clip_input)
    result[mid] = 1.0 + ((arr[mid] - max_input) / (clip_input - max_input)) * (
        max_output - 1.0
    )

    high = valid & (arr >= clip_input)
    result[high] = max_output

    return np.clip(result, 0.0, max_output)


def build_urban_overlay_from_json(payload: dict, flip: bool) -> Optional[str]:
    """Recreate EO Browser Sentinel-1 styling directly from the JSON payload."""
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
    compressed = highlight_compress(
        composite,
        max_input=0.8,
        clip_input=0.9,
        max_output=1.0,
    )
    rgba = np.zeros(composite.shape[:-1] + (4,), dtype=np.uint8)
    rgba[..., :3] = (np.clip(compressed, 0.0, 1.0) * 255).astype(np.uint8)

    valid_mask = mask > 0.5
    rgba[~valid_mask] = 0
    rgba[..., 3][valid_mask] = 200
    if flip:
        rgba = np.flipud(rgba)
    return rgba_to_data_url(rgba)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{value}'.")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def grid_to_rgba(
    grid: np.ndarray,
    vmin: float,
    vmax: float,
    colors: Sequence[str],
    fade_from_transparent: bool = False,
) -> np.ndarray:
    if np.isclose(vmin, vmax):
        raise ValueError("min and max values must differ.")

    norm = (grid - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)

    valid = np.isfinite(grid)
    if not np.any(valid):
        return rgba

    ramp_positions = np.linspace(0.0, 1.0, len(colors))
    ramp_rgb = np.array([_hex_to_rgb(color) for color in colors], dtype=np.float32)
    norm_valid = norm[valid]
    for channel in range(3):
        channel_values = np.interp(norm_valid, ramp_positions, ramp_rgb[:, channel])
        rgba[..., channel][valid] = channel_values.astype(np.uint8)
    if fade_from_transparent:
        alpha = (norm_valid * 200).astype(np.uint8)
    else:
        alpha = np.full(norm_valid.shape, 200, dtype=np.uint8)
    rgba[..., 3][valid] = alpha
    return rgba


def rgba_to_data_url(rgba: np.ndarray) -> str:
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass
class OverlaySpec:
    name: str
    bbox: Sequence[float]
    grid: np.ndarray | None = None
    min_value: float | None = None
    max_value: float | None = None
    colors: Sequence[str] | None = None
    image_data_url: str | None = None
    show_colormap: bool = True
    opacity: float = 0.8
    fade_from_transparent: bool = False


def add_overlay(fmap: folium.Map, spec: OverlaySpec) -> None:
    min_lon, min_lat, max_lon, max_lat = spec.bbox
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    if spec.image_data_url is not None:
        image_url = spec.image_data_url
    elif spec.grid is not None:
        if spec.min_value is None or spec.max_value is None or spec.colors is None:
            raise ValueError(f"Grid overlay '{spec.name}' missing range or colors.")
        rgba = grid_to_rgba(
            spec.grid,
            spec.min_value,
            spec.max_value,
            spec.colors,
            fade_from_transparent=spec.fade_from_transparent,
        )
        image_url = rgba_to_data_url(rgba)
    else:
        raise ValueError(f"Overlay '{spec.name}' has no image source.")

    folium.raster_layers.ImageOverlay(
        name=spec.name,
        image=image_url,
        bounds=bounds,
        opacity=spec.opacity,
        interactive=False,
        cross_origin=False,
    ).add_to(fmap)

    if spec.show_colormap and spec.grid is not None and spec.colors is not None:
        colormap = LinearColormap(
            colors=list(spec.colors),
            vmin=spec.min_value,
            vmax=spec.max_value,
        )
        colormap.caption = spec.name
        colormap.add_to(fmap)


def create_map(overlays: Sequence[OverlaySpec], zoom: int, output_path: Path) -> None:
    if not overlays:
        raise ValueError("No overlays were provided.")

    min_lon, min_lat, max_lon, max_lat = overlays[0].bbox
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    fmap = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    for overlay in overlays:
        add_overlay(fmap, overlay)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(output_path)
    print(f"Saved {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cloud/urban/aurora overlay map.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use live NOAA aurora probabilities instead of simulated data.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    cloud_payload, cloud_grid = load_grid(CLOUD_JSON, CLOUD_BAND_KEY)
    cloud_grid = orient_grid(cloud_grid, GRID_ORIGIN)

    urban_payload, urban_grid = load_grid(URBAN_JSON, URBAN_BAND_KEY)
    urban_grid = orient_grid(urban_grid, GRID_ORIGIN)

    if cloud_payload.get("bbox") != urban_payload.get("bbox"):
        raise ValueError("Cloud and urban datasets must share an identical bbox.")

    no2_payload = no2_grid = None
    if NO2_JSON.exists():
        no2_payload, no2_grid = load_grid(NO2_JSON, NO2_BAND_KEY)
        no2_grid = orient_grid(no2_grid, GRID_ORIGIN)
        if no2_payload.get("bbox") != cloud_payload.get("bbox"):
            raise ValueError("NO2 dataset bbox does not match the cloud dataset.")
    else:
        print(f"NO2 file '{NO2_JSON}' not found; skipping NO2 overlay.")

    target_bbox = cloud_payload["bbox"]
    flip_urban = GRID_ORIGIN == "south_up"

    urban_image = build_urban_overlay_from_json(urban_payload, flip=flip_urban)
    overlays = [
        OverlaySpec(
            name="Cloud optical thickness",
            bbox=target_bbox,
            grid=cloud_grid,
            min_value=CLOUD_MIN,
            max_value=CLOUD_MAX,
            colors=CLOUD_COLORS,
        ),
    ]

    if urban_image is not None:
        overlays.append(
            OverlaySpec(
                name="Urban density (EO style)",
                bbox=target_bbox,
                image_data_url=urban_image,
                show_colormap=False,
                opacity=0.8,
            )
        )
    else:
        overlays.append(
            OverlaySpec(
                name="Urban density",
                bbox=target_bbox,
                grid=urban_grid,
                min_value=URBAN_MIN,
                max_value=URBAN_MAX,
                colors=URBAN_COLORS,
            )
    )

    if INCLUDE_AURORA:
        try:
            aurora_payload, aurora_grid = load_aurora_overlay(args.real)
        except Exception as exc:  # noqa: BLE001 - surface network/fetch issues
            print(f"Failed to load aurora overlay: {exc}")
        else:
            name = AURORA_NAME if not args.real else "NOAA aurora probability"
            overlays.append(
                OverlaySpec(
                    name=name,
                    bbox=aurora_payload["bbox"],
                    grid=np.flipud(aurora_grid),
                    min_value=AURORA_MIN,
                    max_value=AURORA_MAX,
                    colors=AURORA_COLORS,
                    fade_from_transparent=True,
                )
            )

    if no2_payload and no2_grid is not None:
        overlays.append(
            OverlaySpec(
                name="NO₂ concentration",
                bbox=target_bbox,
                grid=no2_grid,
                min_value=NO2_MIN,
                max_value=NO2_MAX,
                colors=NO2_COLORS,
            )
        )

    create_map(overlays, MAP_ZOOM, OUTPUT_HTML)


if __name__ == "__main__":
    main()
