import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from util import label_map

# Load saved model
model = load_model("model.h5")


img_path = "00009.png"  # Your image path
img = image.load_img(img_path, target_size=(30, 30))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Predict
prediction = model.predict(img_tensor)

if prediction.ndim > 1:
    predicted_index = np.argmax(prediction, axis=1)[0] # Get the index of the highest probability along axis 1 (for each sample) and take the first one
else:
    predicted_index = np.argmax(prediction) # Get the index of the highest probability

predicted_label = predicted_index + 10
if predicted_label in label_map:
    predicted_label_name = label_map[predicted_label]
    print(f"Predicted: {predicted_label}:{predicted_label_name}")
else:
    print(f"Predicted Label Value: {predicted_label} - Label name not found in the mapping.")