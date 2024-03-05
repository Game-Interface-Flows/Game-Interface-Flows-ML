from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

app = FastAPI()


@app.get("/health")
def get_home():
    """Check API status"""
    status = {"health_check": "OK"}
    return JSONResponse(content=jsonable_encoder(status))
