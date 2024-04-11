import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import UploadFile
from PIL import Image
from tqdm import tqdm


class Screen:
    def __init__(self, image_index: int, similarty):
        self.image_index = image_index
        self.similarty = similarty


class TimedScreen:
    def __init__(self, image_index: int, time: int):
        self.image_index = image_index
        self.time = time


class PredictionService:
    def __init__(
        self, model_path: str, screen_prob_threshold: float, screen_sim_threshold: float
    ):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.screen_prob_threshold = screen_prob_threshold
        self.screen_sim_threshold = screen_sim_threshold

    @staticmethod
    def _get_image_hist(image_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([image], [0], None, [512], [0, 512])
        return hist

    @staticmethod
    def _score_image_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score

    @staticmethod
    def _preprocess_image(image_bytes: bytes) -> Image:
        image = Image.open(io.BytesIO(image_bytes))
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return transform(image).unsqueeze(0)

    async def _process_image(self, image, executor: ThreadPoolExecutor):
        bytes = await image.read()
        loop = asyncio.get_event_loop()
        model_image = await loop.run_in_executor(
            executor, self._preprocess_image, bytes
        )
        image_hist = await loop.run_in_executor(executor, self._get_image_hist, bytes)
        return model_image, image_hist

    async def _get_processed_images(self, images: List[UploadFile]):
        with ThreadPoolExecutor() as executor:
            tasks = [self._process_image(image, executor) for image in images]
            processed_images = await asyncio.gather(*tasks)
        return processed_images

    async def get_screens_flow(
        self, images: List[UploadFile], images_interval: int = 1
    ) -> List[TimedScreen]:
        processed_images = await self._get_processed_images(images)
        screens = set()
        times_screens = []
        progress_bar = tqdm(total=len(processed_images))
        for curr_index, (model_image, sim_image) in enumerate(processed_images):
            progress_bar.update(1)
            # check if screen is already stored
            curr_screen = None
            for screen in screens:
                sim = self._score_image_similarity(sim_image, screen.similarty)
                if sim > self.screen_sim_threshold:
                    curr_screen = screen
                    break

            # if image does not exist, we should create a screen
            if curr_screen is None:
                with torch.no_grad():
                    prob = self.model(model_image).item()
                    pred = prob > self.screen_prob_threshold
                if pred == 0:
                    continue
                curr_screen = Screen(curr_index, sim_image)
                screens.add(curr_screen)

            timed_screen = TimedScreen(
                curr_index * images_interval, curr_screen.image_index
            )
            times_screens.append(timed_screen)

        progress_bar.close()
        return times_screens


MODEL_PATH = "api/model.pt"
CLASS_PROB_TRESHOLD = 0.5
IMG_SIM_TRESHOLD = 0.85
prediction_service = PredictionService(
    MODEL_PATH, CLASS_PROB_TRESHOLD, IMG_SIM_TRESHOLD
)
