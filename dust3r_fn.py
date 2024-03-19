import os
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    print(f' (Found {len(imgs)} images)')
    return imgs


def load_panorama_pairs(xyz_metadatas, headings=(0, 1, 2, 3), symmetry=False):
    pid2idx = {meta["pano_id"] : idx for idx, meta in enumerate(xyz_metadatas)}
    pano_images = dict()
    image_id = 0
    
    for idx, meta in enumerate(xyz_metadatas):
        panoid = meta["pano_id"]
        images = load_images([f'./Street/{panoid}/gsv_{hid}.jpg' for hid in headings], size=512)
        for idx in range(len(images)):
            images[idx]["idx"] = image_id
            images[idx]["instance"] = str(image_id)
            image_id += 1
        
        pano_images[panoid] = images
    
    connected_components = [[]]
    visited = set()
    
    
    # Identify connected components in the implicit graph.
    while len(visited) < len(pano_images):
        for init_seed, meta in enumerate(xyz_metadatas):
            if meta["pano_id"] not in visited: break
        fringe  = {xyz_metadatas[init_seed]["pano_id"]}
        
        while len(fringe) > 0:
            pid = fringe.pop()
            if pid in visited: continue
            visited.add(pid)
            
            images = pano_images[pid]
            
            # Add intra-panorama pair
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    connected_components[-1].append((images[i], images[j]))
            
            # Add inter-panorama pair
            for nid in xyz_metadatas[pid2idx[pid]]["neighbors"]:
                if nid not in pid2idx: continue # input graph may be partial.
                
                fringe.add(nid)
                
                # To avoid (i, j), (j, i) duplication.
                if pid2idx[nid] < pid2idx[pid]:
                    continue
                
                neighbor_images = pano_images[nid]
                for i in range(len(images)):
                    for j in range(len(neighbor_images)):
                        connected_components[-1].append((images[i], neighbor_images[j]))
        
        if symmetry:
            reverse_pairs = []
            for i, j in connected_components[-1]: reverse_pairs.append((j, i))
            connected_components[-1].extend(reverse_pairs)
        
        connected_components.append([])
    return connected_components
