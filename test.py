import math
import json
import requests

# Helper to compute a bounding box from the centre (lat, lon) and zoom
def bounding_box_from_zoom(lat, lon, zoom, pixel_size=512, tile_size=256):
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

# centre and zoom from your map URL
bbox = bounding_box_from_zoom(60.184960, 24.875736, zoom=6)

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
    input: ["CLOUD_OPTICAL_THICKNESS", "dataMask"],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.CLOUD_OPTICAL_THICKNESS, sample.dataMask];
}
"""

process_payload = {
    "input": {
        "bounds": {
            "bbox": bbox,
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
        },
        "data": [{
            "type": "sentinel-5p-l2",
            "dataFilter": {
                "timeRange": {
                    "from": "2025-11-06T00:00:00Z",
                    "to":   "2025-11-06T23:59:59Z"
                }
            }
        }]
    },
    "output": {
        "width": 512,
        "height": 512,
        "responses": [{
            "identifier": "default",
            "format": {"type": "image/tiff"}
        }]
    },
    "evalscript": evalscript
}

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}
process_url = 'https://sh.dataspace.copernicus.eu/api/v1/process'
resp = requests.post(process_url, headers=headers, json=process_payload)
resp.raise_for_status()  # will raise HTTPError if request failed

# Save the GeoTIFF
with open('CLOUD_OPTICAL_THICKNESS.tif', 'wb') as f:
    f.write(resp.content)
print('Saved CLOUD_OPTICAL_THICKNESS.tif')
