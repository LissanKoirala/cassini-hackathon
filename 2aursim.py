from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

CLOUD_JSON = Path("CLOUD_OPTICAL_THICKNESS.json")
OUTPUT_JSON = Path("AURORA_SIMULATED_2.json")
GRID_ROWS = 500
GRID_COLS = 800
MAX_PROBABILITY = 100.0


def simulate_auroral_oval() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic auroral oval over the northern hemisphere."""
    lat_values = np.linspace(50, 85, GRID_ROWS)  # typical auroral latitudes
    lon_values = np.linspace(-180, 180, GRID_COLS)
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

    # Auroral oval parameters (approx.)
    oval_center_lat = 67.0
    oval_width = 4.5  # latitude width of the band
    oval_intensity_peak = 1.0

    # Gaussian distribution around center latitude
    lat_component = np.exp(-((lat_grid - oval_center_lat) ** 2) / (2 * oval_width**2))

    # Longitudinal modulation: stronger near midnight (lon ~ 0–30E) and weaker near noon (~180E)
    local_time_factor = np.cos(np.radians(lon_grid)) ** 2
    modulated = lat_component * local_time_factor

    # Add noise and variability
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, size=modulated.shape)
    probability = np.clip((modulated + noise) * MAX_PROBABILITY, 0.0, MAX_PROBABILITY)

    return lat_values, lon_values, probability


def build_coordinate_list(
    lat_values: np.ndarray, lon_values: np.ndarray, probability_grid: np.ndarray
) -> list[list[float]]:
    coords = []
    for i, lat in enumerate(lat_values):
        for j, lon in enumerate(lon_values):
            value = float(probability_grid[i, j])
            if value > 0.01:  # skip zero-value points for smaller output
                coords.append([float(lon), float(lat), round(value, 3)])
    return coords


def main() -> None:
    lat_values, lon_values, probabilities = simulate_auroral_oval()
    coords = build_coordinate_list(lat_values, lon_values, probabilities)

    now = datetime.now(timezone.utc)
    bbox = [-180.0, 50.0, 180.0, 85.0]
    payload = {
        "Observation Time": now.isoformat(),
        "Forecast Time": (now + timedelta(hours=1)).isoformat(),
        "coordinates": coords,
        "bbox": bbox,
        "description": "Simulated northern auroral oval probability field",
    }

    OUTPUT_JSON.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Generated auroral oval with {len(coords)} points covering northern hemisphere.")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
