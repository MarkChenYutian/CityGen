import torch
from PIL import Image

import torch
import numpy as np
from lang_sam.lang_sam import (
    SAM_MODELS,
    sam_model_registry,
    SamPredictor,
    load_model_hf,
    transform_image,
    predict,
    box_ops
)

class LangSAMInterface():

    def __init__(self, sam_type="vit_h", ckpt_path=None, device="cuda"):
        self.sam_type = sam_type
        self.device = device
        self.build_groundingdino()
        self.build_sam(ckpt_path)

    def build_sam(self, ckpt_path):
        if self.sam_type is None or ckpt_path is None:
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            checkpoint_url = SAM_MODELS[self.sam_type]
            try:
                sam = sam_model_registry[self.sam_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            except:
                raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                    and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                    re-downloading it.")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        else:
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except:
                raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
                should match your checkpoint path: {ckpt_path}. Recommend calling LangSAM \
                using matching model type AND checkpoint path")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits



class LangSAMExtractor:
    def __init__(self, textprompt: str, modeltype: str, modelckpt: str, min_size: int, device: str) -> None:
        super().__init__()
        self.textprompt = textprompt
        self.device = device
        self.model = LangSAMInterface(modeltype, modelckpt, device)
        self.min_size = min_size
    
    @torch.no_grad()
    @torch.inference_mode()
    def segment(self, img: Image):
        masks, boxes, phrases, logits = self.model.predict(img, self.textprompt)
        confident_masks = masks[logits > 0.5]
        if confident_masks.size(0) == 0:
            out = torch.zeros(1, img.size[1], img.size[0], dtype=torch.float, device=self.device)
        else:
            out = (torch.sum(confident_masks, dim=0, keepdim=True) > 0).float()
        
        # Morphology operation - dilate with kernelsize=5s
        out = torch.nn.functional.max_pool2d(out, kernel_size=7, stride=1, padding=3).bool()
        return out.unsqueeze(0)
