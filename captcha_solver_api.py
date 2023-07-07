from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from model.predict_captcha import predict
from contextlib import asynccontextmanager
import pickle
from keras.models import load_model
from keras.backend import clear_session

MODEL_FILENAME = "model/captcha_model.keras"
MODEL_LABELS_FILENAME = "model/model_labels.dat"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        app.state.lb = pickle.load(f)

    # Load the trained neural network
    app.state.model = load_model(MODEL_FILENAME)
    yield
    clear_session()


app = FastAPI(title="Captcha Solver API", lifespan=lifespan)


class Img(BaseModel):
    img_url: str


@app.post("/predict/captcha/", status_code=200)
async def predict_captcha(file: UploadFile = File(...)):
    img = await file.read()
    prediction = predict(img, app.state.model, app.state.lb)
    if not prediction:
        raise HTTPException(
            status_code=501, detail="Couldn't predict captcha:("
        )
    return prediction
