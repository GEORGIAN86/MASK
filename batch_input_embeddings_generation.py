from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import torch
import json
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, dtype=torch.float32, device=device)
    return image.permute(2, 0, 1).contiguous()

sam_checkpoint = "/home/sumit/segment-anything/Model/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

RESIZE_TRANSFORM = ResizeLongestSide(sam.image_encoder.img_size)

true_image_dir = "/home/sumit/segment-anything/output/IMAGES"
image_files = sorted(list(Path(true_image_dir).glob("*.jpg")))

embedding_dir = "encoder_embeddings"
os.makedirs(embedding_dir, exist_ok=True)

batch_size = 8
batch_images = []
batch_names = []

for image_path in tqdm(image_files, total=len(image_files), desc="Processing Images in Batches"):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    prepared = prepare_image(image, RESIZE_TRANSFORM, device)
    batch_images.append(prepared)
    batch_names.append(image_name)

    if len(batch_images) == batch_size or image_path == image_files[-1]:
        with torch.no_grad():
            input_batch = torch.stack(batch_images)
            embeddings = sam.image_encoder(input_batch)

        for i, name in enumerate(batch_names):
            embedding_np = embeddings[i].cpu().numpy()
            np.save(os.path.join(embedding_dir, f"{name}_embedding.npy"), embedding_np)

        del input_batch
        del embeddings
        torch.cuda.empty_cache()

        batch_images.clear()
        batch_names.clear()
