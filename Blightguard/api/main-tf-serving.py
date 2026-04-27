from http.client import responses

from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import numpy as np
from PIL import Image
import uvicorn
from pygments.lexers import data
import requests
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from rich import json
from starlette import endpoints

#from tensorflow.python.grappler.item import Item
#from tensorflow.python.keras.engine.training_v1 import Model

app = FastAPI()

origins = [
    "https://blightguard-rosy.vercel.app",
    "https://blight-guard-saud-sayyeds-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight","Late Blight", "Healthy"]
@app.get("/")
async def ping():
    return "Hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(file: UploadFile = File(...)
 ):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    prediction = response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
