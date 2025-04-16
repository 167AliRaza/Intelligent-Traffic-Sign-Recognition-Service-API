from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
from io import BytesIO
from util import label_map


app = FastAPI()
try:
    model = load_model("model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = image.resize((30, 30))
        image_array = np.array(image)
        tensor = np.expand_dims(image_array, axis=0)
        tensor = tensor.astype(np.float32) / 255.0
        prediction = model.predict(tensor)
         
        if prediction.ndim > 1:
             predicted_index = np.argmax(prediction, axis=1)[0] # Get the index of the highest probability along axis 1 (for each sample) and take the first one
        else:
             predicted_index = np.argmax(prediction) # Get the index of the highest probability
        predicted_label = predicted_index + 10
        if predicted_label in label_map:
            predicted_label_name = label_map[predicted_label]
            print(f"Predicted label: {predicted_label}, Label name: {predicted_label_name}")
            return { "predicted_label": int(predicted_label), "predicted_label_name": predicted_label_name }	
    
        else:
            return { "predicted_label": predicted_label, "message": "Label name not found in the mapping." }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
