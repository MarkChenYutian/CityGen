import pickle
import util.visualize as vis
from rebuild_fn import load_panorama_pairs
from interface_dust3r import local_reconstruct


# https://www.openstreetmap.org/
# ^^ I found this to be very useful to find where you want to visualize!
# I selected a small block near CMU since I know this place : )
LONGTITUDE_RNG = [-79.96, -79.95]
LATITUDE_RNG   = [ 40.44,  40.445]

if __name__ == "__main__":
    import rerun as rr
    rr.init("Application", spawn=True)
    
    with open("pittsburgh_pano_graph.pkl", "rb") as fb:
        metas = pickle.load(fb)
    
    # vis.visualize_pano_location(LONGTITUDE_RNG, LATITUDE_RNG, metas)
    
    # pair_groups = load_panorama_pairs([metas[i] for i in [1, 2, 51, 76, 77]], "./Data/Street_6view", headings=(0, 1, 2, 3, 4, 5), symmetry=True)
    # pair_groups = load_panorama_pairs([metas[i] for i in [0, 50]], "./Data/Street_6view", headings=(0, 1, 2, 3, 4, 5), symmetry=True)
    pair_groups = load_panorama_pairs([metas[i] for i in [0]], "./Data/Street_6view", headings=(0, 1, 2, 3, 4, 5), symmetry=True)
    for group in pair_groups:
        if len(group) <= 1: continue
        vis.visualize_ptcloud(*local_reconstruct(group))
