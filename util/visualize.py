import torch
import rerun as rr
import numpy as np

# For GIS visualization
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from PIL import Image
from pathlib import Path
from itertools import product

def visualize_ptcloud(pts_arr: list[torch.Tensor], img_arr: list[torch.Tensor], poses: torch.Tensor):
    rr.log("world/points", rr.Points3D(
        torch.cat([pts.reshape(-1, 3) for pts in pts_arr], dim=0),
        colors=torch.cat([img.reshape(-1, 3) for img in img_arr], dim=0)
    ))
    
    for i in range(poses.size(0)):
        rr.log(f"world/cam/{i}",
                rr.Pinhole(resolution=(512, 384), focal_length=(320, 320)),
                rr.Transform3D(mat3x3=poses[i, :3, :3], translation=poses[i, :3, 3])
        )


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
    pid2idx = {meta["pano_id"] : idx for idx, meta in enumerate(xyz_metadatas)}
    
    locs = np.array([meta["location"]["xyz"] for meta in xyz_metadatas])
    xy_locs = locs[..., :2]
    
    ax: plt.Axes = plt.axes()
    ax.scatter(locs[..., 0], locs[..., 1], s=0.5)
    
    lines = []
    for idx, meta in enumerate(xyz_metadatas):
        if "neighbors" not in meta: continue
        
        for nid in meta["neighbors"]:
            lines.append((xy_locs[idx], xy_locs[pid2idx[nid]]))
    ax.add_collection(LineCollection(lines))
    
    plt.show()
