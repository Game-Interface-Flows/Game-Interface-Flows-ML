from pydantic import BaseModel


class Prediction(BaseModel):
    index: int
    prediction: int
    time_in: int
    time_out: int


class CustomStatus(BaseModel):
    status_name: str
    status_code: str
