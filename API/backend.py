import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import requests

app=FastAPI()

# Command to start Deep learning Server
# docker run -t --rm -p 8506:8501 -v D:/Formal/TensorflowProject/potato_disease_project:/potato_disease_project tensorflow/serving --rest_api_port=8501 --model_config_file=/potato_disease_project/API/models.config

endpoint="http://localhost:8506/v1/models/potato_model:predict"

CLASS_NAMES=['Early_blight', 'Late_blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "I am Alive..."

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile =File(...)):
    # print(type(file))
    image = read_file_as_image(await file.read())
    img_array = np.expand_dims(image, 0)

    json_data = {
        "instances": img_array.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction=np.array(response.json()["predictions"][0])

    predicted_class=CLASS_NAMES[np.argmax(prediction)]
    confidence=np.max(prediction)

    return {
        "Class": predicted_class,
        "Confidence": confidence
    }

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)