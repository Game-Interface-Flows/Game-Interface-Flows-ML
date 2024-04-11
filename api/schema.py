from pydantic import BaseModel


class PredictedScreen(BaseModel):
    index: int
    time_in: int
    time_out: int


class CustomStatus(BaseModel):
    status_name: str
    status_code: str
