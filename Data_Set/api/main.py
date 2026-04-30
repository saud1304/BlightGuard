from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "https://blightguard-rosy.vercel.app",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model("./models/1")
    return MODEL

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def home():
    return {"message": "API is running"}

@app.get("/test-model")
def test_model():
    try:
        get_model()
        return {"status": "model loaded successfully"}
    except Exception as e:
        return {"error": str(e)}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB").resize((256, 256))
    return np.array(image)
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())

        img_batch = np.expand_dims(image, 0).astype("float32")

        predictions = MODEL(img_batch)
        predictions = list(predictions.values())[0].numpy()

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }

    except Exception as e:
        return {"error": str(e)}
