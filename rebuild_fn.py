import os
import torch
import PIL.Image
import numpy as np
import pypose as pp
import torchvision.transforms as tvf

from PIL.ImageOps import exif_transpose
from typing_extensions import NamedTuple

from clean_frame import LangSAMExtractor

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
DynamicMask_Vehicle = LangSAMExtractor("vehicle", "vit_l", "/data2/datasets/yutianch/SpaceQA/models/sam_vit_l_0b3195.pth", 40, "cuda")
DynamicMask_Person  = LangSAMExtractor("person", "vit_l", "/data2/datasets/yutianch/SpaceQA/models/sam_vit_l_0b3195.pth", 40, "cuda")


class PairGroup(NamedTuple):
    recon_pairs: list
    world_xyz: list


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def load_images(folder_or_list, size, square_ok=False):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.JPG']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        # Remove Google copyright watermark
        img = img.crop((0, 0, W1, H1 - 30))
        # Re-establish image size
        W1, H1 = img.size
        
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        
        mask1 = DynamicMask_Vehicle.segment(img)
        mask2 = DynamicMask_Person.segment(img)
        mask = torch.logical_or(mask1, mask2)[0]
        
        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(
            img=ImgNorm(img)[None],
            mask=mask,
            true_shape=np.int32([img.size[::-1]]),
            idx=len(imgs),
            instance=str(len(imgs))
        ))

    assert imgs, 'no images foud at '+root
    print(f' (Found {len(imgs)} images)')
    return imgs

def load_panorama_pairs(xyz_metadatas, pano_path, headings=(0, 1, 2, 3), symmetry=False) -> list[PairGroup]:
    pid2idx = {meta["pano_id"] : idx for idx, meta in enumerate(xyz_metadatas)}
    pano_images = dict()
    image_id = 0
    
    for idx, meta in enumerate(xyz_metadatas):
        panoid = meta["pano_id"]
        images = load_images([f'{pano_path}/{panoid}/gsv_{hid}.jpg' for hid in headings], size=512)
        for idx in range(len(images)):
            images[idx]["idx"] = image_id
            images[idx]["instance"] = str(image_id)
            images[idx]["xyz"] = torch.tensor(meta["location"]["xyz"])
            image_id += 1
        
        pano_images[panoid] = images
    
    connected_components = []
    visited = set()
    
    
    # Identify connected components in the implicit graph.
    while len(visited) < len(pano_images):
        idx, cc, world_xyz = 0, [], []
        
        for init_seed, meta in enumerate(xyz_metadatas):
            if meta["pano_id"] not in visited: break
        fringe  = {xyz_metadatas[init_seed]["pano_id"]}
        # world_xyz = xyz_metadatas[init_seed]["location"]["xyz"]
        
        
        while len(fringe) > 0:
            pid = fringe.pop()
            if pid in visited: continue
            visited.add(pid)
            
            images = pano_images[pid]
            
            # Re-generate index for grouped dust3r optimization
            for image in images:
                image["idx"] = idx
                image["instance"] = str(idx)
                idx += 1
            world_xyz.append(images[0]["xyz"])
            
            # Add intra-panorama pair
            for curr_idx in range(len(images)):
                left_idx = curr_idx - 1
                cc.append((images[curr_idx], images[left_idx], torch.tensor([0., 0., 0.])))
            
            # Add inter-panorama pair
            for nid in xyz_metadatas[pid2idx[pid]]["neighbors"]:
                if nid not in pid2idx: continue # input graph may be partial.
                
                fringe.add(nid)
                
                # To avoid (i, j), (j, i) duplication. Will add in "if symmetry" step if required.
                if pid2idx[nid] < pid2idx[pid]:
                    continue
                
                neighbor_images = pano_images[nid]
                for i in range(len(images)):
                    for j in range(len(neighbor_images)):
                        cc.append((images[i], neighbor_images[j], neighbor_images[j]["xyz"] - images[i]["xyz"]))
        
        if symmetry:
            reverse_pairs = []
            for i, j, pos in cc: reverse_pairs.append((j, i, -1*pos))
            cc.extend(reverse_pairs)
        
        connected_components.append(PairGroup(recon_pairs=cc, world_xyz=world_xyz))
    return connected_components

def to_camera_coordinate(poses, points):
    poses = pp.from_matrix(poses, pp.SE3_type)
    T_d2c: pp.LieTensor = poses[0].Inv()
    poses = T_d2c @ poses
    points = [T_d2c.Act(pc) for pc in points]
    return poses, points

def scale_alignment_world(poses, points, world_xyzs, num_cam_per_pano):
    assert len(world_xyzs) > 1, "Mathematically impossible to reconstruct scale with one xyz"
    assert poses.size(0) > num_cam_per_pano, "Mathematically impossible to reconstruct scale with one pano"
    
    current_scale = (poses[0].translation() - poses[num_cam_per_pano].translation()).norm()
    world_scale = (world_xyzs[0] - world_xyzs[1]).norm()
    
    delta_scale = world_scale / current_scale
    poses[..., :3] *= delta_scale
    
    return poses, [point * delta_scale for point in points]


def rotateA2B(a, b):
    # Step 1: Axis of rotation (v) via cross product
    v = torch.linalg.cross(a, b)
    v_norm = torch.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm  # Normalize if not a zero vector

    # Step 2: Cosine of rotation angle using dot product
    cos_theta = torch.dot(a, b)

    # Step 3: Construct skew-symmetric matrix V from v
    V = torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Using Rodrigues' rotation formula: R = I + sin(theta)*V + (1-cos(theta))*V^2
    # Since we don't directly have sin(theta), we use the identity sin^2(theta) = 1 - cos^2(theta) to compute sin(theta)
    sin_theta = torch.sqrt(1 - cos_theta**2)
    I = torch.eye(3)
    R = I + sin_theta * V + (1 - cos_theta) * V @ V
    return R


def to_world_coordinate(poses, points, world_xyzs, num_cam_per_pano):
    assert len(world_xyzs) > 1, "Mathematically impossible to align with one xyz (since we need to align rotation)"
    assert poses.size(0) > num_cam_per_pano, "Mathematically impossible to align with one pano (since we need to align rotation)"
    
    ## NEU -> EDN
    displacement = torch.stack(world_xyzs).to(poses.device)
    displacement_edn = torch.stack([displacement[..., 1], -1 * displacement[..., 2], displacement[..., 0]], dim=1)
    frame0_ref = displacement_edn[0]
    frame1_ref = displacement_edn[1]
    ##
    
    frame0_pos = poses[:num_cam_per_pano].translation().mean(dim=0)
    frame1_pos = poses[num_cam_per_pano:num_cam_per_pano * 2].translation().mean(dim=0)
    
    # Rotation Alignment (very confusing...)
    unit_est = frame1_pos - frame0_pos
    unit_ref = frame1_ref - frame0_ref
    unit_est = unit_est / torch.linalg.norm(unit_est)
    unit_ref = unit_ref / torch.linalg.norm(unit_ref)
    
    _R = rotateA2B(unit_est.double().cpu(), unit_ref.double().cpu()).to(poses.device)
    R: pp.LieTensor = pp.from_matrix(_R, pp.SE3_type)
        
    points = [R.Act(pc - frame0_pos.unsqueeze(0).double()) + frame0_ref.unsqueeze(0).double() for pc in points]
    poses[..., :3] -= frame0_pos
    new_poses = R.float() @ poses
    new_poses[..., :3] += frame0_ref

    return new_poses, points
