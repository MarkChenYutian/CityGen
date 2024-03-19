import torch
from util.context import AddPath
from itertools import product

with AddPath("./dust3r/"):
    from dust3r.inference import inference, load_model
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

MODEL_PATH = "D:/Data/TDW/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DEVICE = 'cuda'
BATCH_SIZE = 1
SCHEDULE = 'cosine'
LR = 0.01
N_ITER = 300

model = load_model(MODEL_PATH, DEVICE)


def local_reconstruct(pairs):
    # headings = [0, 1, 2, 3]
    # views = product(pano_ids, headings)
    
    # images = load_images([f'./Street/{pid}/gsv_{hid}.jpg' for pid, hid in views], size=512)
    
    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, DEVICE, batch_size=BATCH_SIZE)

    scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
    
    # Remove the "Google" copyright watermark by setting confidence sufficiently low
    # for idx in range(len(scene.im_conf)):
    #     scene.im_conf[idx].data[360:] = 1.
    
    loss = scene.compute_global_alignment(init="mst", niter=N_ITER, schedule=SCHEDULE, lr=LR)

    pts3ds = scene.get_pts3d()
    masks  = scene.get_masks()
    imgs = scene.imgs
    
    return [p[m].detach().cpu() for p, m in zip(pts3ds, masks)], [torch.tensor(img[m.detach().cpu()]) for img, m in zip(imgs, masks)]
