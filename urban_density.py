import math
import json
from io import BytesIO

import numpy as np
import requests
import tifffile
from PIL import Image


MAP_LAT = 67.127155
MAP_LON = 25.510257
LEAFLET_ZOOM = 6
# Keep the full bbox aligned with the EO Browser tile (512 px) for reference.
BBOX_PIXEL_SIZE = 512
# Split the bbox into smaller segments and sample each at the maximum resolution.
TILES_PER_AXIS = 2
OUTPUT_SIZE = 2500
BAND_NAME = "URBAN_DENSITY"


def bounding_box_from_zoom(lat, lon, zoom, pixel_size=512, tile_size=256):
    """Compute bbox similar to leaflet zoom level."""
    deg_per_pix = 360.0 / (tile_size * 2**zoom)
    width_deg = deg_per_pix * pixel_size
    min_lon = lon - width_deg / 2.0
    max_lon = lon + width_deg / 2.0

    def lat2mercator_y(lat_deg):
        lat_rad = math.radians(lat_deg)
        return (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2

    def mercator_y2lat(y):
        n = math.pi * (1 - 2 * y)
        return math.degrees(math.atan(math.sinh(n)))

    y = lat2mercator_y(lat)
    delta = pixel_size / (tile_size * 2**zoom)
    max_lat = mercator_y2lat(y - delta / 2.0)
    min_lat = mercator_y2lat(y + delta / 2.0)
    return [min_lon, min_lat, max_lon, max_lat]


def split_bbox(full_bbox, tiles_per_axis):
    """Split the full bbox into a grid of smaller bounding boxes."""
    min_lon, min_lat, max_lon, max_lat = full_bbox
    lon_edges = np.linspace(min_lon, max_lon, tiles_per_axis + 1)
    lat_edges = np.linspace(min_lat, max_lat, tiles_per_axis + 1)
    grid = []
    for row in range(tiles_per_axis):
        row_tiles = []
        lat_min = lat_edges[tiles_per_axis - row - 1]
        lat_max = lat_edges[tiles_per_axis - row]
        for col in range(tiles_per_axis):
            lon_min = lon_edges[col]
            lon_max = lon_edges[col + 1]
            row_tiles.append([lon_min, lat_min, lon_max, lat_max])
        grid.append(row_tiles)
    return grid


FULL_BBOX = bounding_box_from_zoom(MAP_LAT, MAP_LON, zoom=LEAFLET_ZOOM, pixel_size=BBOX_PIXEL_SIZE)

client_id = 'sh-fe913274-f231-4fba-b8f7-2a5d668e178e'
client_secret = 'ALuP8cxWven92rhxm8DUSUO2SmJZGHMl'

token_url = ('https://identity.dataspace.copernicus.eu/auth/realms/CDSE/'
             'protocol/openid-connect/token')
token_data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret
}
token_resp = requests.post(token_url, data=token_data)
token_resp.raise_for_status()
access_token = token_resp.json()['access_token']

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask"],
    output: [
      { id: "default", bands: 4 },
      { id: "analysis", bands: 4, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  const vh = sample.VH;
  const vv = sample.VV;
  const urbanScore = vh * 5.5;
  const isUrban = urbanScore > 0.5 ? 1 : 0;
  const vhBlue = vh * 8.0;

  return {
    default: [isUrban, vv, vhBlue, sample.dataMask],
    analysis: [isUrban, urbanScore, vh, vv],
    dataMask: [sample.dataMask]
  };
}
"""

CRS = "http://www.opengis.net/def/crs/EPSG/0/4326"
DATA_SOURCE = [{
    "type": "sentinel-1-grd",
    "dataFilter": {
        "timeRange": {
            "from": "2025-11-01T00:00:00Z",
            "to": "2025-11-30T23:59:59Z"
        },
        "acquisitionMode": "IW",
        "polarization": "DV",
        "resolution": "HIGH"
    },
    "processing": {
        "backCoeff": "SIGMA0_ELLIPSOID"
    }
}]

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}
process_url = 'https://sh.dataspace.copernicus.eu/api/v1/process'


def fetch_response(
    identifier: str,
    format_type: str,
    width: int = OUTPUT_SIZE,
    height: int = OUTPUT_SIZE,
    bbox_override=None,
) -> requests.Response:
    bounds = {
        "bbox": bbox_override if bbox_override is not None else FULL_BBOX,
        "properties": {"crs": CRS}
    }
    payload = {
        "input": {
            "bounds": bounds,
            "data": DATA_SOURCE,
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{
                "identifier": identifier,
                "format": {"type": format_type}
            }]
        },
        "evalscript": evalscript
    }
    response = requests.post(process_url, headers=headers, json=payload)
    if not response.ok:
        raise RuntimeError(
            f"Process API '{identifier}' failed with {response.status_code}: {response.text}"
        )
    return response


def read_single_band_tiff(content: bytes, label: str) -> np.ndarray:
    try:
        array = tifffile.imread(BytesIO(content))
    except (tifffile.TiffFileError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to decode {label} response as TIFF (bytes={len(content)})."
        ) from exc

    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            raise RuntimeError(
                f"Unexpected {label} image shape {array.shape}; expecting single band."
            )
    if array.ndim != 2:
        raise RuntimeError(
            f"Unexpected {label} image shape {array.shape}; expecting 2D array."
        )
    return array


def read_multi_band_tiff(content: bytes, label: str) -> np.ndarray:
    try:
        array = tifffile.imread(BytesIO(content))
    except (tifffile.TiffFileError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to decode {label} response as TIFF (bytes={len(content)})."
        ) from exc

    if array.ndim == 2:
        raise RuntimeError(
            f"Unexpected {label} image shape {array.shape}; expecting multi-band array."
        )
    if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) and array.shape[-1] not in (1, 2, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise RuntimeError(
            f"Unexpected {label} image shape {array.shape}; expecting 3D array."
        )
    return array


visual_path = f"{BAND_NAME}_visual.png"
tile_size = OUTPUT_SIZE
tiles = TILES_PER_AXIS
full_size = tiles * tile_size

visual_mosaic = np.zeros((full_size, full_size, 4), dtype=np.float32)
analysis_mosaic = np.zeros((full_size, full_size, 4), dtype=np.float32)
mask_mosaic = np.zeros((full_size, full_size), dtype=np.float32)

tile_bboxes = split_bbox(FULL_BBOX, tiles)

for row_idx, row_tiles in enumerate(tile_bboxes):
    for col_idx, tile_bbox in enumerate(row_tiles):
        row_slice = slice(row_idx * tile_size, (row_idx + 1) * tile_size)
        col_slice = slice(col_idx * tile_size, (col_idx + 1) * tile_size)

        default_resp = fetch_response(
            "default", "image/tiff", bbox_override=tile_bbox
        )
        analysis_resp = fetch_response(
            "analysis", "image/tiff", bbox_override=tile_bbox
        )
        mask_resp = fetch_response(
            "dataMask", "image/tiff", bbox_override=tile_bbox
        )

        default_tile = read_multi_band_tiff(default_resp.content, "default").astype(np.float32)
        analysis_tile = read_multi_band_tiff(analysis_resp.content, "analysis").astype(np.float32)
        mask_tile = read_single_band_tiff(mask_resp.content, "dataMask").astype(np.float32)

        visual_mosaic[row_slice, col_slice, :] = default_tile
        analysis_mosaic[row_slice, col_slice, :] = analysis_tile
        mask_mosaic[row_slice, col_slice] = mask_tile

visual_image = Image.fromarray(
    np.clip(visual_mosaic, 0, 255).astype(np.uint8), mode="RGBA"
)
visual_image.save(visual_path)


analysis_array = analysis_mosaic
mask_array = mask_mosaic

if analysis_array.shape[:2] != mask_array.shape:
    raise RuntimeError(
        f"Shape mismatch between analysis {analysis_array.shape} and dataMask {mask_array.shape}."
    )

mask_norm = np.where(mask_array > 0, 1.0, 0.0).astype(np.float32)
urban_density = analysis_array[..., 0].astype(np.float32)
urban_score = analysis_array[..., 1].astype(np.float32)
vh_backscatter = analysis_array[..., 2].astype(np.float32)
vv_backscatter = analysis_array[..., 3].astype(np.float32)

for layer in (urban_density, urban_score, vh_backscatter, vv_backscatter):
    layer[mask_norm < 0.5] = np.nan

valid_pixels = int(np.count_nonzero(mask_norm))
urban_pixels = int(np.nansum(np.where(np.isnan(urban_density), 0, urban_density)))
urban_percentage = (
    float(urban_pixels) / valid_pixels * 100 if valid_pixels else 0.0
)

result = {
    "bbox": FULL_BBOX,
    "width": int(urban_density.shape[1]),
    "height": int(urban_density.shape[0]),
    "urban_density": urban_density.tolist(),
    "urban_score": urban_score.tolist(),
    "vh_backscatter": vh_backscatter.tolist(),
    "vv_backscatter": vv_backscatter.tolist(),
    "data_mask": mask_norm.tolist(),
    "summary": {
        "valid_pixels": valid_pixels,
        "urban_pixels": urban_pixels,
        "urban_percentage": urban_percentage,
        "vh_mean": float(np.nanmean(vh_backscatter)),
        "vv_mean": float(np.nanmean(vv_backscatter)),
    }
}

with open(f'{BAND_NAME}.json', 'w', encoding='utf-8') as f:
    json.dump(result, f)

print(
    f"Saved {visual_path} and {BAND_NAME}.json "
    f"(urban coverage ~{urban_percentage:.2f}% of {valid_pixels} valid pixels)"
)
