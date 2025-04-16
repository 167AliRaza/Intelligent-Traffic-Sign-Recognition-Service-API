import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil

# Load saved model
model = load_model("model.h5")

# Create output directories if they don't exist
label_map = {
    0: '10', 1: '11', 2: '12', 3: '13', 4: '14', 5: '15', 6: '16', 7: '17', 8: '18', 9: '19',
    10: '20', 11: '21', 12: '22', 13: '23', 14: '24', 15: '25', 16: '26', 17: '27', 18: '28', 19: '29',
    20: '30', 21: '31', 22: '32', 23: '33', 24: '34', 25: '35', 26: '36', 27: '37', 28: '38', 29: '39',
    30: '40', 31: '41', 32: '42'
}

# Create output directories if they don't exist
output_root_dir = "predicted_classes"
for label_value in label_map.values():
    os.makedirs(os.path.join(output_root_dir, label_value), exist_ok=True)
# Folder containing images
data_folder = "dataset/test"

# Iterate over images in data folder
for img_file in os.listdir(data_folder):
    img_path = os.path.join(data_folder, img_file)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(30, 30))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    # Predict
    prediction = model.predict(img_tensor)
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_map[predicted_class_index]
    
    destination_folder = os.path.join(output_root_dir, predicted_label)
    destination_path = os.path.join(destination_folder, img_file)
    # Move file based on prediction
    
    shutil.move(img_path, destination_path)
    print(f"Moved '{img_file}' to '{destination_folder}'")
