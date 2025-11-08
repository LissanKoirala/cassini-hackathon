from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

CLOUD_JSON = Path("CLOUD_OPTICAL_THICKNESS.json")
OUTPUT_JSON = Path("AURORA_SIMULATED.json")
GRID_ROWS = 120
GRID_COLS = 120
MAX_PROBABILITY = 100.0


def load_bbox(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Reference file '{path}' not found.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    bbox = payload.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError(f"File '{path}' missing bbox metadata.")
    return bbox


def simulate_probability_field(bbox: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_values = np.linspace(min_lon, max_lon, GRID_COLS)
    lat_values = np.linspace(min_lat, max_lat, GRID_ROWS)

    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

    center_lat = max_lat - 0.2 * (max_lat - min_lat)
    center_lon = min_lon + 0.55 * (max_lon - min_lon)

    lat_std = 0.25 * (max_lat - min_lat)
    lon_std = 0.35 * (max_lon - min_lon)

    gauss = np.exp(
        -(((lat_grid - center_lat) ** 2) / (2 * lat_std**2))
        - (((lon_grid - center_lon) ** 2) / (2 * lon_std**2))
    )

    gradient = np.clip((lat_grid - min_lat) / (max_lat - min_lat), 0.0, 1.0)

    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, 0.08, size=gauss.shape)

    probability = (0.6 * gauss + 0.3 * gradient + noise).clip(0.0, 1.0) * MAX_PROBABILITY
    return lat_values, lon_values, probability


def build_coordinate_list(
    lat_values: np.ndarray, lon_values: np.ndarray, probability_grid: np.ndarray
) -> list[list[float]]:
    coords = []
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            value = float(probability_grid[i, j])
            coords.append([float(lon), float(lat), round(value, 3)])
    return coords


def main() -> None:
    bbox = load_bbox(CLOUD_JSON)
    lat_values, lon_values, probabilities = simulate_probability_field(bbox)
    coords = build_coordinate_list(lat_values, lon_values, probabilities)

    now = datetime.now(timezone.utc)
    payload = {
        "Observation Time": now.isoformat(),
        "Forecast Time": (now + timedelta(hours=1)).isoformat(),
        "coordinates": coords,
        "bbox": bbox,
    }
    OUTPUT_JSON.write_text(json.dumps(payload), encoding="utf-8")

    print(
        f"Generated simulated aurora field with {len(coords)} points "
        f"({GRID_ROWS}x{GRID_COLS}) covering bbox {bbox}."
    )
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
