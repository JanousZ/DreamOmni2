import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch

class Replace5kDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            self.image_dirs = json.load(f)
        self.read_mode = "PIL"
        self.is_image_preprocess = True

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        return self.get_sample(index)
    
    def preprocess(self, img):
        # img: numpy, [H, W, C], uint8
        img = img.astype(np.float32) / 127.5 - 1.0  # -> [-1,1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> [C,H,W]
        return img

    def get_sample(self, idx):
        image_dir = self.image_dirs[idx]
        ref_image_path = os.path.join(image_dir,'reference_image.png')
        src_image_path = os.path.join(image_dir,'raw_image.png')
        gt_image_path = os.path.join(image_dir,'ground_truth.png')

        if self.read_mode == "cv2":
            gt_image = cv2.imread(gt_image_path)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            ref_image = cv2.imread(ref_image_path)
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            src_image = cv2.imread(src_image_path)
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        
        if self.read_mode == "PIL":
            gt_image = Image.open(gt_image_path).convert("RGB")
            ref_image = Image.open(ref_image_path).convert("RGB")
            src_image = Image.open(src_image_path).convert("RGB")
        
        instruction = None

        if self.is_image_preprocess:
            gt_image = self.preprocess(gt_image)
            ref_image = self.preprocess(ref_image)
            src_image = self.preprocess(src_image)
        
        item = {
            "src_image_path": src_image_path,
            "ref_image_path": ref_image_path,
            "tgt_image_path": gt_image_path,
            "src_image": src_image,
            "ref_image": ref_image,
            "tgt_image": gt_image,
            "prompt": instruction
        }

        return item