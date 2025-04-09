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

def extract_centroids(mask, image=None, save_name=None, save_vis=False, vis_output_dir=None):
    if mask is None:
        return []

    if len(mask.shape) == 2:  
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_bgr = mask

    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []  
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    if save_vis and image is not None and save_name is not None and vis_output_dir is not None:
        image_vis = image.copy()
        for cx, cy in centroids:
            cv2.circle(image_vis, (cx, cy), 5, (0, 0, 255), -1)
        output_path = os.path.join(vis_output_dir, save_name)
        cv2.imwrite(output_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))

    return centroids

def compute_iou(org_mask, pred_mask):
    pred_mask = pred_mask.astype(bool)
    intersection = np.logical_and(org_mask, pred_mask)
    union = np.logical_or(org_mask, pred_mask)
    if union.sum() == 0:
        return 0.0
    return intersection.sum() / union.sum()

def part_mask(original_mask, centroid):
    centroid = centroid.squeeze()
    
    if len(original_mask.shape) == 3:
        gray = cv2.cvtColor(original_mask, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_mask.copy()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(gray)

    pt = (float(centroid[0]), float(centroid[1]))

    for cnt in contours:        
        if cv2.pointPolygonTest(cnt, pt, False) >= 0:
            cv2.drawContours(new_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            break

    return new_mask

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)
RESIZE_TRANSFORM = ResizeLongestSide(sam.image_encoder.img_size)

true_image_dir = "/mnt/mydisk/CrescerAi/Bhumi/Segment_anything/IMAGES"
ground_truth_dir = "/mnt/mydisk/CrescerAi/Bhumi/Segment_anything/MASKS"

image_files = sorted(list(Path(true_image_dir).glob("*.jpg")))
mask_files = sorted(list(Path(ground_truth_dir).glob("*.png")))

sam_input_images = []
image_and_its_mask = {}
json_data = []

for image_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing Images"):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_name = os.path.splitext(os.path.basename(mask_path))[0]

    if image_name == mask_name:
        print(image_name)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_input_image = prepare_image(image, RESIZE_TRANSFORM, device=sam.device)
        sam_input_images.append(sam_input_image)

        org_mask = cv2.imread(str(mask_path))
        org_mask_rgb = cv2.cvtColor(org_mask, cv2.COLOR_BGR2RGB)

        centroids = extract_centroids(org_mask)
        centroids_tensor = torch.tensor(centroids, device=sam.device)
        if centroids_tensor.ndim == 1:
            centroids_tensor = centroids_tensor.unsqueeze(0)

        sam_input_points = torch.tensor(centroids, dtype=torch.float, device=sam.device)
        resized_centroids = RESIZE_TRANSFORM.apply_coords_torch(centroids_tensor, original_size=image.shape[:2])

        predictor.set_image(image) 
        
        all_ious = []
        for centroid_ in centroids_tensor:
            input_label = np.ones(1, dtype=int)
            input_point = centroid_.cpu().numpy().reshape(1, 2)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            original_mask_gt = part_mask(original_mask=org_mask_rgb , centroid=centroid_)
            iou = compute_iou(original_mask_gt , masks[0])
            all_ious.append(iou)

            image_and_its_mask[image_path.name] = masks

        jj = {
            'image': image_name,
            'centroids': centroids,
            'iou': all_ious
        }
        json_data.append(jj)

with open("images_and_ious.json", "w") as f:
    json.dump(json_data, f, indent=4)
