import torch
from util.context import AddPath

with AddPath("./dust3r/"):
    from dust3r.inference import inference, load_model
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

MODEL_PATH = "/data2/datasets/yutianch/StreetView/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DEVICE = 'cuda'
BATCH_SIZE = 1
SCHEDULE = 'linear'
LR = 0.01
N_ITER = 1000
MIN_CONF_THR = 6

model = load_model(MODEL_PATH, DEVICE)


def local_reconstruct(pairs_with_gps):
    pairs = [(i, j) for i, j, _ in pairs_with_gps]
    ref_pairwise_pos = [p for _, _, p in pairs_with_gps]
    
    output = inference(pairs, model, DEVICE, batch_size=BATCH_SIZE)

    scene = global_aligner(
        output,
        device=DEVICE,
        mode=GlobalAlignerMode.GPSInformedOptimizer,
        ref_pairwise_pos=ref_pairwise_pos,
        min_conf_thr=MIN_CONF_THR
    )
    
    loss = scene.compute_global_alignment(init="mst", niter=N_ITER, schedule=SCHEDULE, lr=LR)

    pts3ds = scene.get_pts3d()
    masks  = scene.get_masks()
    imgs = scene.imgs
    
    points = [p[m].detach().cpu() for p, m in zip(pts3ds, masks)]
    colors = [torch.tensor(img[m.detach().cpu()]) for img, m in zip(imgs, masks)]
    poses  = scene.get_im_poses().detach().cpu()
    return points, colors, poses
