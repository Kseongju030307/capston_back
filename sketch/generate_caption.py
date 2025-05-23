import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import cv2
import numpy as np
from .preprocessing import preprocess_sketch

# BASE_DIR = 프로젝트 최상단 (manage.py 있는 위치)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 모델 경로
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov11_best.pt')
CLIP_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'clip.pth')
CATEGORY_FILE_PATH = os.path.join(BASE_DIR, 'models', 'dataset_category.txt')

# ===== Adapter 정의 =====
class Adapter(nn.Module):
    def __init__(self, c_in=512):
        super().__init__()
        self.fc1 = nn.Linear(c_in, c_in, bias=True)  # ✅ bias 켬
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c_in, c_in, bias=True)  # ✅ bias 켬
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


def image_crop(image_path):
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO 모델 파일이 존재하지 않음: {YOLO_MODEL_PATH}")

    model = YOLO(YOLO_MODEL_PATH)
    image = cv2.imread(image_path)
    results = model(image)[0]

    images = []
    for idx, box in enumerate(results.boxes):
        conf = float(box.conf[0])
        if conf < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        temp_input_path = os.path.join(BASE_DIR, 'media', f'crop_raw_{idx}.png')
        temp_output_path = os.path.join(BASE_DIR, 'media', f'crop_processed_{idx}.png')

        Image.fromarray(cropped_rgb).save(temp_input_path)
        preprocess_sketch(temp_input_path, temp_output_path)

        processed_image = Image.open(temp_output_path).convert("RGB")
        images.append(processed_image)

    return images

def image_classification(images):
    if not os.path.exists(CLIP_MODEL_PATH):
        raise FileNotFoundError(f"CLIP 모델 파일이 존재하지 않음: {CLIP_MODEL_PATH}")
    if not os.path.exists(CATEGORY_FILE_PATH):
        raise FileNotFoundError(f"카테고리 파일이 존재하지 않음: {CATEGORY_FILE_PATH}")

    with open(CATEGORY_FILE_PATH, 'r') as f:
        categories = [line.strip() for line in f.readlines()]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    adapter = Adapter(c_in=512).to(device).eval()
    adapter.load_state_dict(torch.load(CLIP_MODEL_PATH, map_location=device))

    text_inputs = processor(
        text=[f"a photo of a {c}" for c in categories],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    predictions = []

    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            image_feat = clip_model.get_image_features(**inputs)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

            adapted_feat = image_feat + adapter(image_feat)
            adapted_feat = adapted_feat / adapted_feat.norm(dim=-1, keepdim=True)

            probs = torch.softmax(adapted_feat @ text_features.T, dim=-1).cpu().squeeze()
            pred_idx = probs.argmax().item()
            predictions.append(categories[pred_idx])

    return predictions