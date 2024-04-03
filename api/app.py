import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import api.config as config
from api.schema import *

import cv2

model = torch.jit.load(config.MODEL_PATH)
model.eval()
app = FastAPI()

model_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

sim_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)


def preprocess_image(image_bytes, transform):
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def preprocess_image_2(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return image.resize(size=(512, 512))


def get_image_hist(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [512], [0, 512])
    return hist

def image_similarity_opencv(hist1, hist2):
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

def image_similarity(img1, img2):
    return ssim(np.array(img1), np.array(img2), multichannel=True, win_size=3)


@app.get("/health")
def get_health() -> CustomStatus:
    """Check API status"""
    return CustomStatus(status_name="health", status_code="OK")


async def process_frame(frame, executor):
    frame_bytes = await frame.read()
    loop = asyncio.get_event_loop()
    model_image = await loop.run_in_executor(
        executor, preprocess_image, frame_bytes, model_transform
    )
    #sim_image = await loop.run_in_executor(executor, preprocess_image_2, frame_bytes)
    image_hist = await loop.run_in_executor(executor, get_image_hist, frame_bytes)
    return model_image, image_hist


@app.post("/flow")
async def get_direct_flow(frames: List[UploadFile] = File(...)) -> List[Prediction]:
    predictions = []
    image_tensors = {}
    image_probs = {}

    with ThreadPoolExecutor() as executor:
        tasks = [process_frame(frame, executor) for frame in frames]
        processed_frames = await asyncio.gather(*tasks)

    progress_bar = tqdm(total=len(processed_frames))
    for i, (model_image, sim_image) in enumerate(processed_frames):
        index = i

        for k in image_tensors:
            sim = image_similarity_opencv(sim_image, image_tensors[k])
            if sim > config.IMG_SIM_TRESHOLD:
                index = k
                break

        if index != i:
            prob = image_probs[index]
        else:
            with torch.no_grad():
                prob = model(model_image).item()
                pred = prob > config.CLASS_PROB_TRESHOLD

        if pred == 0:
            continue

        if index == i:
            image_tensors[index] = sim_image
            image_probs[index] = prob

        if len(predictions) > 0 and predictions[-1].index == index:
            predictions[-1].time_out += config.SECONDS_INTERVAL
        else:
            prediction = Prediction(
                prediction=pred,
                time_in=i * config.SECONDS_INTERVAL,
                time_out=(i + 1) * config.SECONDS_INTERVAL,
                index=index,
            )
            predictions.append(prediction)
        progress_bar.update(1)

    progress_bar.close()

    return predictions
