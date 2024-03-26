import torch
import pickle
import util.visualize as vis
from rebuild_fn import load_panorama_pairs, to_camera_coordinate, scale_alignment_world, to_world_coordinate
from interface_dust3r import local_reconstruct

def flatten(xs):
    acc = []
    for x in xs: acc.extend(x)
    return acc

# https://www.openstreetmap.org/
# ^^ I found this to be very useful to find where you want to visualize!
# I selected a small block near CMU since I know this place : )
LONGTITUDE_RNG = [-79.96, -79.95]
LATITUDE_RNG   = [ 40.44,  40.445]

if __name__ == "__main__":
    import rerun as rr
    rr.init("Application", spawn=True)
    rr.save("/data/yutianch/StreetView/pointcloud.rrd")
    
    with open("pittsburgh_pano_graph.pkl", "rb") as fb:
        metas = pickle.load(fb)
    
    # vis.visualize_pano_location(LONGTITUDE_RNG, LATITUDE_RNG, metas)
    
    panoidx = [0, 50, 76, 77, 94, 116]
    headings = (0, 1, 2, 3, 4, 5)
    # panoidx = [0, 50]
    pair_groups = load_panorama_pairs([metas[i] for i in panoidx], "./Data/Street_6view", headings=headings, symmetry=True)
    recon_groups = []
    for idx, group in enumerate(pair_groups):
        if len(group.recon_pairs) <= 1: continue
        print(f"Processing group {idx}..., world_coord {group.world_xyz}")
        points, colors, confs, poses = local_reconstruct(group.recon_pairs)
        
        # Coordinate frame correction (dust3r random frame -> camera 0 frame -> world frame)
        # Assumption: Camera 0 always face same direction (according to Google's API doc, cam 0 = heading 0deg always
        # faces north)
        poses, points = to_camera_coordinate(poses, points)
        poses, points = scale_alignment_world(poses, points, group.world_xyz, len(headings))
        poses, points = to_world_coordinate(poses, points, group.world_xyz, len(headings))
        
        recon_groups.append((points, colors, confs, poses))
    
    all_points = flatten([p for p, _, _, _ in recon_groups])
    all_colors = flatten([c for _, c, _, _ in recon_groups])
    all_poses  = torch.cat([p for _, _, _, p in recon_groups], dim=0)
    vis.visualize_ptcloud(all_points, all_colors, all_poses)
    vis.visualize_ref_pos(torch.stack([torch.tensor(metas[i]["location"]["xyz"]) for i in panoidx]))
