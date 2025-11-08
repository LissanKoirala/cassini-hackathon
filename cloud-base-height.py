
"""
Request Sentinel Hub CLOUD_BASE_HEIGHT data using the Copernicus EO Browser
Evalscript provided by the user and save the outputs locally.
"""

from __future__ import annotations

import io
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests


# --- Configuration ---------------------------------------------------------

# Defaults can be overridden via environment variables to keep secrets safe.
CLIENT_ID = os.getenv("CDSE_CLIENT_ID", "sh-fe913274-f231-4fba-b8f7-2a5d668e178e")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET", "ALuP8cxWven92rhxm8DUSUO2SmJZGHMl")

# Bounding box derived from map centre + zoom (same approach as in test.py).
MAP_LAT = float(os.getenv("CBH_MAP_LAT", "60.184960"))
MAP_LON = float(os.getenv("CBH_MAP_LON", "24.875736"))
MAP_ZOOM = int(os.getenv("CBH_MAP_ZOOM", "6"))

# Time range for Sentinel-5P Level 2 data.
TIME_FROM = os.getenv("CBH_TIME_FROM", "2025-11-06T00:00:00Z")
TIME_TO = os.getenv("CBH_TIME_TO", "2025-11-06T23:59:59Z")

# Target dataset settings.
DATASET_TYPE = os.getenv("CBH_DATASET", "sentinel-5p-l2")
CRS = "http://www.opengis.net/def/crs/EPSG/0/4326"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

# Evalscript provided by the user (EO Browser -> Custom script).
EVALSCRIPT = """//VERSION=3
var minVal = 0.0;
var maxVal = 20000.0;

function setup() {
  return {
    input: ["CLOUD_BASE_HEIGHT", "dataMask"],
    output: [
      {
        id: "default",
        bands: 4,
      },
      {
        id: "index",
        bands: 1,
        sampleType: "FLOAT32" 
      },
      {
        id: "eobrowserStats",
        bands: 1,
      },
      {
        id: "dataMask",
        bands: 1
      },
    ],
  };
}

var viz = ColorRampVisualizer.createBlueRed(minVal, maxVal);

function evaluatePixel(samples) {
  let [r, g, b] = viz.process(samples.CLOUD_BASE_HEIGHT);

  const statsVal = isFinite(samples.CLOUD_BASE_HEIGHT) ? samples.CLOUD_BASE_HEIGHT : NaN;
  return {
    default: [r, g, b, samples.dataMask],
    index: [samples.CLOUD_BASE_HEIGHT],
    eobrowserStats: [statsVal],
    dataMask: [samples.dataMask],
  };
}
"""

# Define which outputs we want to persist locally.
OUTPUT_SPECS = [
    ("default", {"type": "image/tiff"}, Path("CLOUD_BASE_HEIGHT_RGBA.tif")),
    ("index", {"type": "image/tiff"}, Path("CLOUD_BASE_HEIGHT_INDEX.tif")),
    ("dataMask", {"type": "image/tiff"}, Path("CLOUD_BASE_HEIGHT_MASK.tif")),
]

# Optional: EO Browser stats are numeric; save as JSON.
OUTPUT_SPECS.append(
    ("eobrowserStats", {"type": "application/json"}, Path("CLOUD_BASE_HEIGHT_STATS.json"))
)


# --- Helpers ---------------------------------------------------------------

def bounding_box_from_zoom(
    lat: float,
    lon: float,
    zoom: int,
    pixel_size: int = 512,
    tile_size: int = 256,
) -> Tuple[float, float, float, float]:
    """Compute bounding box centered on (lat, lon) matching EO Browser tiles."""

    deg_per_pix = 360.0 / (tile_size * 2**zoom)
    width_deg = deg_per_pix * pixel_size
    min_lon = lon - width_deg / 2.0
    max_lon = lon + width_deg / 2.0

    def lat2mercator_y(lat_deg: float) -> float:
        lat_rad = math.radians(lat_deg)
        return (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2

    def mercator_y2lat(y: float) -> float:
        n = math.pi * (1 - 2 * y)
        return math.degrees(math.atan(math.sinh(n)))

    y = lat2mercator_y(lat)
    delta = pixel_size / (tile_size * 2**zoom)
    max_lat = mercator_y2lat(y - delta / 2.0)
    min_lat = mercator_y2lat(y + delta / 2.0)
    return min_lon, min_lat, max_lon, max_lat


def get_access_token(client_id: str, client_secret: str) -> str:
    """Retrieve an access token using client_credentials."""

    token_data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    response = requests.post(TOKEN_URL, data=token_data, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def build_process_payload(bbox: Iterable[float]) -> Dict[str, object]:
    """Create the JSON payload for the Process API call."""

    return {
        "input": {
            "bounds": {"bbox": list(bbox), "properties": {"crs": CRS}},
            "data": [
                {
                    "type": DATASET_TYPE,
                    "dataFilter": {
                        "timeRange": {
                            "from": TIME_FROM,
                            "to": TIME_TO,
                        }
                    },
                }
            ],
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [
                {"identifier": identifier, "format": format_spec}
                for identifier, format_spec, _ in OUTPUT_SPECS
            ],
        },
        "evalscript": EVALSCRIPT,
    }


def save_process_response(content: bytes, content_type: str) -> None:
    """
    Persist API response(s). Multiple outputs are returned as a ZIP archive,
    while a single response arrives as the raw file. The helper handles both.
    """

    if content_type.startswith("application/zip") or content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.infolist():
                target = Path(member.filename).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"Saved {target} (from zip)")
        return

    # When we only asked for a single response, map it back to its filename.
    if len(OUTPUT_SPECS) == 1:
        target_path = OUTPUT_SPECS[0][2]
    else:
        # Fallback: generic name when API unexpectedly returns single payload.
        target_path = Path("CLOUD_BASE_HEIGHT_response.bin")

    with open(target_path, "wb") as file:
        file.write(content)
    print(f"Saved {target_path}")


def main() -> None:
    bbox = bounding_box_from_zoom(MAP_LAT, MAP_LON, MAP_ZOOM)
    print(f"Requesting CLOUD_BASE_HEIGHT for bbox={bbox}")

    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = build_process_payload(bbox)
    response = requests.post(
        PROCESS_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=90,
    )
    response.raise_for_status()
    save_process_response(response.content, response.headers.get("content-type", ""))


if __name__ == "__main__":
    main()
