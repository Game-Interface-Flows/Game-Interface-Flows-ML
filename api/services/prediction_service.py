import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

from api.utils.load_model import load_model


class Screen:
    def __init__(self, image_index: int, similarity):
        self.image_index = image_index
        self.similarity = similarity


class TimedScreen:
    def __init__(self, image_index: int, time: int):
        self.image_index = image_index
        self.time = time


class PredictionService:
    def __init__(
        self,
        model_weights: str,
        screen_prob_threshold: float,
        screen_sim_threshold: float,
    ):
        self.model = load_model(num_classes=2, model_weights=model_weights)
        self.model.eval()
        self.screen_prob_threshold = screen_prob_threshold
        self.screen_sim_threshold = screen_sim_threshold

    @staticmethod
    def base64_to_numpy(base64_string: str) -> np.array:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    
    @staticmethod
    def _preprocess_image_for_ssim(image_np: np.ndarray):
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image = resize(image, (256, 256), anti_aliasing=True)
        return image

    @staticmethod
    def _preprocess_image_for_model(image_np: np.array) -> torch.Tensor:
        transform = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.ToGray(always_apply=True),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ]
        )
        transformed = transform(image=image_np)["image"]
        transformed = transformed.unsqueeze(0)
        return transformed

    async def _preprocess_image(self, image_np: np.array, executor: ThreadPoolExecutor):
        loop = asyncio.get_event_loop()
        model_image = await loop.run_in_executor(
            executor, self._preprocess_image_for_model, image_np
        )
        ssim_image = await loop.run_in_executor(
            executor, self._preprocess_image_for_ssim, image_np
        )
        return model_image, ssim_image

    async def _get_processed_images(self, images: List[np.array]):
        with ThreadPoolExecutor() as executor:
            tasks = [self._preprocess_image(image, executor) for image in images]
            processed_images = await asyncio.gather(*tasks)
        return processed_images

    async def get_screens_flow(
        self, encoded_images: List[str], images_interval: int = 3
    ) -> List[TimedScreen]:
        decoded_images = [self.base64_to_numpy(encoded) for encoded in encoded_images]
        processed_images = await self._get_processed_images(decoded_images)
        screens = set()
        timed_screens = []
        for curr_index, (model_image, ssim_image) in enumerate(processed_images):
            # check if screen is already stored
            curr_screen = None
            for screen in screens:
                sim = ssim(ssim_image, screen.similarity, data_range=1.0)
                if sim > self.screen_sim_threshold:
                    curr_screen = screen
                    break

            # if image does not exist, we should create a screen
            if curr_screen is None:
                with torch.no_grad():
                    logits = self.model(model_image)
                    probabilities = F.softmax(logits, dim=1)
                    probability = (probabilities[0, 1]).item()
                    pred = probability > self.screen_prob_threshold
                if pred == 0:
                    continue
                curr_screen = Screen(curr_index, ssim_image)
                screens.add(curr_screen)

            timed_screen = TimedScreen(
                time = curr_index * images_interval,
                image_index = curr_screen.image_index
            )
            timed_screens.append(timed_screen)

        return timed_screens


MODEL_WEIGHTS = "api/services/resnet18_weights.pth"
CLASS_PROB_TRESHOLD = 0.95
IMG_SIM_TRESHOLD = 0.5
prediction_service = PredictionService(
    MODEL_WEIGHTS, screen_prob_threshold=CLASS_PROB_TRESHOLD, screen_sim_threshold=IMG_SIM_TRESHOLD
)
