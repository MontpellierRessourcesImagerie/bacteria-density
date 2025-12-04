import numpy as np
from shapely.geometry import Polygon, Point
from rasterio import features

def as_polygon(coordinates):
    xy = coordinates[..., -2:][..., ::-1]
    return Polygon(xy)

def make_crop(image, polygon):
    H, W = image.shape[-2:]
    mask2d = features.rasterize(
        [(polygon, 1)],
        out_shape=(H, W),
        fill=0,
        dtype=image.dtype
    )
    maskNd = mask2d[np.newaxis, ...] * image
    minx, miny, maxx, maxy = polygon.bounds
    cropped = maskNd[:, int(miny):int(maxy), int(minx):int(maxx)]
    return cropped

if __name__ == "__main__":
    from pathlib import Path
    import tifffile

    image_path = Path("/home/clement/Documents/projects/2119-bacteria-density/small-data/nuclei.tif")
    stack = tifffile.imread(image_path)
    coordinates = np.array([
        [  11.     ,  984.3795 , 1370.5308 ],
        [  11.     ,  666.43286, 1783.8615 ],
        [  11.     ,  396.17816, 3389.4922 ],
        [  11.     ,  869.9187 , 4066.7185 ],
        [  11.     , 1658.4265 , 4565.895  ],
        [  11.     , 2621.805  , 4578.613  ],
        [  11.     , 2396.0627 , 3548.4656 ],
        [  11.     , 2497.8057 , 2578.728  ],
        [  11.     , 1779.2462 , 2022.3214 ],
        [  11.     , 1944.5785 , 1243.352  ],
        [  11.     , 1543.9657 , 1262.429  ],
        [  11.     , 1356.3771 ,  871.3545 ],
        [  11.     , 1016.1742 , 1046.2252 ],
        [  11.     ,  958.9438 , 1205.1985 ]
    ], dtype=np.float32)

    # p = Point(1000, 1500)
    # print(shape.contains(p))

    cropped = make_crop(stack, coordinates)
    tifffile.imwrite("/tmp/masked_cropped.tif", cropped)