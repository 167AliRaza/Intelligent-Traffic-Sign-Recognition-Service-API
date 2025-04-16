🐾 Cat vs Dog Image Classifier with TensorFlow & FastAPI
A complete deep learning project that trains a Convolutional Neural Network (CNN) to classify images as cats or dogs and deploys it using a FastAPI endpoint for real-time predictions.

📌 Features

✅ Trained on 25,000 labeled images of cats and dogs
✅ TensorFlow/Keras-based CNN with multiple Conv2D and MaxPooling layers
✅ Preprocessing using ImageDataGenerator
✅ Binary classification with sigmoid activation
✅ 20% of the dataset used for validation
✅ FastAPI-based REST API for image classification
✅ Model saved as .h5 for easy deployment and reuse


🛠️ Tech Stack

Python 3.10
TensorFlow/Keras
FastAPI
Uvicorn (ASGI server)
Pillow (for image handling)


📁 Project Structure
cat-dog-classifier/
│
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
│
├── model/
│   └── cat_dog_model.h5          # Trained model
│
├── main.py                       # FastAPI app
├── train_model.py                # Training script
├── pyproject.toml                # Project dependencies
└── README.md                     # This file


🚀 Getting Started
1. Clone the Repository
git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier

2. Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
Install the required packages using the pyproject.toml file:
pip install poetry  # If Poetry is not installed
poetry install

Alternatively, if using a requirements.txt file:
pip install -r requirements.txt

4. Train the Model
Ensure your dataset is organized correctly under the dataset/ folder, then run:
python train_model.py

This will train the model and save it as cat_dog_model.h5 in the model/ directory.
5. Run the API
Start the FastAPI server with Uvicorn:
uvicorn main:app --reload

Visit http://localhost:8000/docs to access the interactive Swagger UI for testing the API.

🖼️ API Usage

Endpoint: POST /predict
Form Field: file (image file, e.g., .jpg, .png)
Returns: Predicted class (Cat or Dog) with confidence score

Example using curl:
curl -X POST \
  -F "file=@your_image.jpg" \
  http://localhost:8000/predict

Example using Python requests:
import requests

with open("your_image.jpg", "rb") as image_file:
    response = requests.post("http://localhost:8000/predict", files={"file": image_file})
print(response.json())


✨ Example Output
{
  "prediction": "Dog",
  "confidence": 0.976
}


🧠 Model Summary

Input Size: 150x150 RGB images
Architecture:
3 Conv2D layers (32, 64, 128 filters)
MaxPooling after each Conv2D layer
Flatten + Dense (512 units)
Dropout (0.5)
Output layer with sigmoid activation




📦 Dependencies
Key dependencies include:

tensorflow
fastapi
uvicorn
pillow
python-multipart (for file uploads in FastAPI)

Full list in pyproject.toml. Install using Poetry or requirements.txt.

📜 License
This project is licensed under the MIT License. Feel free to use and modify it.

🙌 Acknowledgements
Inspired by the classic Kaggle Cats vs Dogs dataset and TensorFlow tutorials.

📣 Let's Connect
If you liked this project or have ideas to improve it, feel free to connect or fork the repo! 🤝
