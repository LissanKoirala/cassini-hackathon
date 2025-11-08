import os
import tifftools

file = "mergedimage.tif"

if os.path.exists(file):
    os.remove(file)

picture1 = tifftools.read_tiff('cloud_base_height.tif')
picture2 = tifftools.read_tiff('CLOUD_OPTICAL_THICKNESS.tif')

picture1['ifds'].extend(picture2['ifds'])
tifftools.write_tiff(picture1, 'mergedimage.tif')
