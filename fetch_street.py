import pickle
import numpy as np
import google_streetview.api
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# For GIS visualization
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from itertools import product

from util.latlng2xyz import latlon_to_xyz, PlateCarree_to_xyz
from util.plotting import matplotlib2numpy


# https://www.openstreetmap.org/
# ^^ I found this to be very useful to find where you want to visualize!
# I selected a small block near CMU since I know this place : )
LONGTITUDE_RNG = [-79.96, -79.95]
LATITUDE_RNG   = [ 40.44,  40.445]

with open("secret.txt", "r") as f: APIKEY = f.read().strip()

def retrieve_patch_pano_info(long_rng: tuple[float, float], lat_rng: tuple[float, float], step_cnt=50):
    long_range = np.linspace(long_rng[0], long_rng[1], num=step_cnt)
    lat_range  = np.linspace(lat_rng[0], lat_rng[1], num=step_cnt)
    
    metadata = []
    pano_ids = set()
    # pts = [pos for pos in product(long_range, lat_range)]
    
    for lat in tqdm(lat_range):
        params = [{
            'size': '512x384', # max 640x640 pixels
            'location': f'{lat},{lng}',
            'heading': '0',
            'pitch': '0',
            'key': APIKEY,
            'return_error_code': 'true',
            'source': 'outdoor',
            'radius': '50'
        }
        for lng in long_range]
        results = google_streetview.api.results(params)
        metas = results.metadata
        
        metas = [m for m in metas if (m["status"] == "OK" and m["pano_id"] not in pano_ids)]
        pano_ids.update({m["pano_id"] for m in metas})
        metadata.extend(metas)
    
    return metadata

def download_panos(panoids: list[str], save_to: str, fov: float=90, headings: tuple[float]=(0, 90, 180, 270)):
    assert Path(save_to).exists(), "Provided path tosave panos does not exist."
    
    for panoid in tqdm(panoids):
        pano_dir = Path(save_to, panoid)
        if pano_dir.exists(): continue  # This panorama is already downloaded!
        
        params = [
            {
                "size": "512x384", 
                "pano": panoid,
                "heading": str(heading),
                "pitch": "0",
                "key": APIKEY,
                "return_error_code": "true",
                "source": "outdoor",
                "fov": str(fov)
            }
            for heading in headings
        ]
        result = google_streetview.api.results(params)
        
        pano_dir.mkdir(parents=True)
        result.download_links(str(pano_dir))

def batch_convert_xyz(metadatas: list) -> None:
    """Convert the location from latitude,longtitude in metadata to a normalized xyz coordinate system.
    
    Args:
        metadatas (list): the metadata for each panorama (streetview)

    Returns:
        None.
        
        The metadata are updated in-place for each panorama. x, y, z are provided in meta["location"]["x"], ...
        separately.
    """
    pos = np.empty((len(metadatas), 3))
    for idx in range(len(metadatas)):
        lat, lng = metadatas[idx]["location"]["lat"], metadatas[idx]["location"]["lng"]
        # x, y, z = latlon_to_xyz(lat, lng)
        x, y, z = PlateCarree_to_xyz(lat, lng, LATITUDE_RNG[0], LONGTITUDE_RNG[0])
        pos[idx, 0] = x
        pos[idx, 1] = y
        pos[idx, 2] = z
    pos -= pos.mean(axis=0, keepdims=True)
    
    for idx in range(len(metadatas)):
        metadatas[idx]["location"]["xyz"] = (pos[idx, 0], pos[idx, 1], pos[idx, 2])

def build_connectivity_graph(xyz_metadatas: list, distance_thresh=0.75):
    locs = np.array([meta["location"]["xyz"] for meta in xyz_metadatas])
    for idx in range(len(xyz_metadatas)):
        dist = np.linalg.norm(locs - locs[idx:idx+1], axis=1)
        nn_dist = min(distance_thresh, dist[dist > 0].min())
        neighbors,  = np.where(dist < nn_dist * 1.25)
        xyz_metadatas[idx]["neighbors"] = neighbors.tolist()
    
def visualize_query_location(lng_rng, lat_rng, step_cnt):
    long_range = np.linspace(lng_rng[0], lng_rng[1], num=step_cnt)
    lat_range  = np.linspace(lat_rng[0], lat_rng[1], num=step_cnt)
    
    request = cimgt.OSM()

    # Bounds: (lon_min, lon_max, lat_min, lat_max):
    extent = [lng_rng[0] - 1e-3, lng_rng[1] + 1e-3, lat_rng[0] - 1e-3, lat_rng[1] + 1e-3]

    ax = plt.axes(projection=request.crs)
    ax.set_extent(extent)
    ax.add_image(request, 18)  # 18 = zoom level, max=20, higher leads to higher resolution.
    
    # Just some random points/lines:
    lats, lngs = [x for _, x in product(long_range, lat_range)], [y for y, _ in product(long_range, lat_range)]
    plt.scatter(lngs, lats, marker="x", transform=ccrs.PlateCarree())
    plt.show()

def visualize_pano_location(long_rng, lat_rng, metadatas: list) -> None:
    request = cimgt.OSM()

    # Bounds: (lon_min, lon_max, lat_min, lat_max):
    extent = [long_rng[0] - 1e-3, long_rng[1] + 1e-3, lat_rng[0] - 1e-3, lat_rng[1] + 1e-3]

    ax = plt.axes(projection=request.crs)
    ax.set_extent(extent)
    ax.add_image(request, 18)  # 18 = zoom level, max=20, higher leads to higher resolution.

    # Just some random points/lines:
    lats, lngs = [], []
    for metadata in metadatas:
        lats.append(metadata["location"]["lat"])
        lngs.append(metadata["location"]["lng"])
    
    plt.scatter(lngs, lats, marker="x", transform=ccrs.PlateCarree())
    plt.show()

def visualize_cam_position(xyz_metadatas: list, image_path: str, min_distance=0.1) -> None:
    import rerun as rr
    
    rr.init("StreetView", spawn=True)
    rr.log("world/", rr.ViewCoordinates(rr.ViewCoordinates.RIGHT_HAND_Z_UP))
    
    visualized_loc = np.zeros((0, 3))
    
    for idx in range(0, len(xyz_metadatas)):
        meta = xyz_metadatas[idx]
        current_loc = np.array(meta["location"]["xyz"])
        
        # Reduce the density of visualization to reduce stress on visualizer.
        # & make it easier to see.
        if visualized_loc.shape[0] > 0:
            dist_to_nn = np.square(np.abs((visualized_loc - current_loc[np.newaxis, ...]))).sum(axis=1).min()
            if dist_to_nn < min_distance: continue
        
        # If visualized, add this to visualized list.
        visualized_loc = np.concatenate([visualized_loc, current_loc[np.newaxis, ...]], axis=0)
        
        rr.log(f"world/Cams/N/{idx}",
               rr.Pinhole(resolution=(512, 384), focal_length=(320, 320), camera_xyz=rr.ViewCoordinates.FRU),
               rr.Image(Image.open(Path(image_path, meta["pano_id"], "gsv_0.jpg"))),
               rr.Transform3D(translation=current_loc)
        )
        
        # Quaternion generated from https://quaternions.online/
        rr.log(f"world/Cams/E/{idx}",
               rr.Pinhole(resolution=(512, 384), focal_length=(320, 320), camera_xyz=rr.ViewCoordinates.FRU),
               rr.Image(Image.open(Path(image_path, meta["pano_id"], "gsv_1.jpg"))),
               rr.Transform3D(translation=meta["location"]["xyz"], rotation=np.array([0., 0., 0.707, 0.707]))
        )
        rr.log(f"world/Cams/S/{idx}",
               rr.Pinhole(resolution=(512, 384), focal_length=(320, 320), camera_xyz=rr.ViewCoordinates.FRU),
               rr.Image(Image.open(Path(image_path, meta["pano_id"], "gsv_2.jpg"))),
               rr.Transform3D(translation=meta["location"]["xyz"], rotation=np.array([0., 0., 1., 0.]))
        )
        rr.log(f"world/Cams/W/{idx}",
               rr.Pinhole(resolution=(512, 384), focal_length=(320, 320), camera_xyz=rr.ViewCoordinates.FRU),
               rr.Image(Image.open(Path(image_path, meta["pano_id"], "gsv_2.jpg"))),
               rr.Transform3D(translation=meta["location"]["xyz"], rotation=np.array([0., 0., 0.707, -0.707]))
        )

def visualize_connectivity_graph(xyz_metadatas: list):
    locs = np.array([meta["location"]["xyz"] for meta in xyz_metadatas])
    xy_locs = locs[..., :2]
    
    ax: plt.Axes = plt.axes()
    ax.scatter(locs[..., 0], locs[..., 1], s=0.5)
    
    lines = []
    for idx, meta in enumerate(xyz_metadatas):
        if "neighbors" not in meta: continue
        
        for neighbor in meta["neighbors"]:
            lines.append((xy_locs[idx], xy_locs[neighbor]))
    ax.add_collection(LineCollection(lines))
    
    plt.show()


if __name__ == "__main__":
    import pdb
    STEPCNT = 50
    
    if Path("pittsburgh_panometa.pkl").exists():
        with open("pittsburgh_panometa.pkl", "rb") as fb:
            metas = pickle.load(fb)
    else:
        metas = retrieve_patch_pano_info(LONGTITUDE_RNG, LATITUDE_RNG, step_cnt=STEPCNT)
        with open("pittsburgh_panometa.pkl", "wb") as fb:
            pickle.dump(metas, fb)
    
    # visualize_query_location(LONGTITUDE_RNG, LATITUDE_RNG, STEPCNT)
    # visualize_pano_location(LONGTITUDE_RNG, LATITUDE_RNG, metas)
    
    pano_ids = [meta["pano_id"] for meta in metas]
    # This will read the cache so won't call expensive GoogleAPI every time.
    download_panos(pano_ids, "./Street")
    
    batch_convert_xyz(metas)
    # visualize_cam_position(metas, "./Street", min_distance=0.2)
    build_connectivity_graph(metas, distance_thresh=0.15)
    
    visualize_connectivity_graph(metas)
    # breakpoint()
    
