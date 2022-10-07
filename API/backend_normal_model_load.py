import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL= tf.keras.models.load_model("..//Models/1")
CLASS_NAMES=['Early_blight', 'Late_blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "I am Alive..."

def read_file_as_image(data) :
    image=np.array(Image.open(BytesIO(data)))
    return image


def predict_image(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    predicted_class, confidence = predict_image(MODEL,image)

    return {
        "Class":predicted_class, 
        "Confidence":confidence
    }

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)