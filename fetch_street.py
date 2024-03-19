import pickle
import numpy as np
import google_streetview.api

from tqdm import tqdm
from pathlib import Path

import util.visualize as vis
from util.latlng2xyz import PlateCarree_to_xyz
from dust3r_fn import load_panorama_pairs
from interface_dust3r import local_reconstruct


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

def batch_convert_xyz(metadatas: list, cvt_fn) -> None:
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
        x, y, z = cvt_fn(lat, lng, LATITUDE_RNG[0], LONGTITUDE_RNG[0])
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
        is_neighbor = dist < nn_dist * 1.25
        is_neighbor[idx] = False
        neighbors,  = np.where(is_neighbor)
        xyz_metadatas[idx]["neighbors"] = [xyz_metadatas[nid]["pano_id"] for nid in neighbors]


if __name__ == "__main__":
    import rerun as rr
    rr.init("Application", spawn=True)
    STEPCNT = 50    # The density of panorama query mesh.
    
    if Path("pittsburgh_panometa.pkl").exists():
        with open("pittsburgh_panometa.pkl", "rb") as fb:
            metas = pickle.load(fb)
    else:
        metas = retrieve_patch_pano_info(LONGTITUDE_RNG, LATITUDE_RNG, step_cnt=STEPCNT)
        with open("pittsburgh_panometa.pkl", "wb") as fb:
            pickle.dump(metas, fb)
    
    # vis.visualize_query_location(LONGTITUDE_RNG, LATITUDE_RNG, STEPCNT)
    # vis.visualize_pano_location(LONGTITUDE_RNG, LATITUDE_RNG, metas)
    
    # This will read the cache so won't call expensive GoogleAPI every time.
    download_panos([meta["pano_id"] for meta in metas], "./Street")
    batch_convert_xyz(metas, cvt_fn=PlateCarree_to_xyz)
    build_connectivity_graph(metas, distance_thresh=0.15)
    
    pair_groups = load_panorama_pairs([metas[i] for i in [1, 2, 51]], symmetry=False)
    # pair_groups = load_panorama_pairs([metas[i] for i in [0, 50]], symmetry=False)
    for group in pair_groups:
        if len(group) == 0: continue
        vis.visualize_ptcloud(*local_reconstruct(group))
    
    # vis.visualize_cam_position(metas, "./Street", min_distance=0.2)
    # vis.visualize_connectivity_graph(metas)
    
    # print(load_panorama_pairs)
    
    # with open("pittsburgh_pano_graph.pkl", "wb") as fb:
    #     pickle.dump(metas, fb)

