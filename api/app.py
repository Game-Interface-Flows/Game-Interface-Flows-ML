from typing import List

from fastapi import FastAPI

from api.schema import *
from api.services.prediction_service import prediction_service

app = FastAPI()


@app.get("/health")
def get_health() -> CustomStatus:
    """Check API status."""
    return CustomStatus(status_name="health", status_code="OK")


@app.post("/flow")
async def get_screens(flow_data: FlowData) -> List[PredictedScreen]:
    """Get screens with time intervals from images."""
    timed_screens = await prediction_service.get_screens_flow(flow_data.encoded_images)
    predicted_screens = []

    for screen in timed_screens:
        if (
            len(predicted_screens) > 0
            and predicted_screens[-1].index == screen.image_index
        ):
            predicted_screens[-1].time_out += flow_data.images_interval
            continue

        screen = PredictedScreen(
            time_in=screen.time,
            time_out=screen.time + flow_data.images_interval,
            index=screen.image_index,
        )
        predicted_screens.append(screen)

    return predicted_screens
