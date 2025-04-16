 # 🚦 Traffic Sign Classification using GTSRB Dataset with FastAPI Integration

This project focuses on classifying German traffic signs using a Convolutional Neural Network (CNN) trained on the [GTSRB dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The trained model is integrated with a FastAPI application for real-time predictions.

---

## 🧠 Model Overview

The model is a CNN built using TensorFlow/Keras and trained on 30x30 resized traffic sign images. It is capable of classifying **33 different traffic sign classes**.

---

## 🔧 Model Architecture

- **Input Layer**: `(30, 30, 3)`
- **Conv2D + ReLU**: 16 filters, 3x3
- **Conv2D + ReLU**: 32 filters, 3x3
- **MaxPooling** + **BatchNormalization**
- **Conv2D + ReLU**: 64 filters
- **Conv2D + ReLU**: 128 filters
- **MaxPooling** + **BatchNormalization**
- **Flatten**
- **Dense**: 512 units + ReLU
- **Dropout**: 0.5
- **Output**: 33 units (softmax)

---

## 🏋️ Training Details

- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Image Size**: 30x30
- **Batch Size**: 32
- **Epochs**: 2
- **Loss Function**: `categorical_crossentropy`
- **Optimizer**: `adam`
- **Final Validation Accuracy**: ✅ `99.7%`

---

## 📁 Dataset Structure
```dataset/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── validation/
    ├── class_0/
    ├── class_1/
    └── ...
```
---

# 🚀 FastAPI Integration
The trained model is served using FastAPI to enable real-time traffic sign prediction from uploaded images.

**🔌 Endpoint**: `/predict`  
**Method**: POST  
**Accepts**: Image file (form-data, field name: `image`)  
**Returns**: Predicted class name and confidence  

**✅ Example Response**  
```json
{
  "predicted_class": "Speed Limit 50 km/h",
  "confidence": 0.97
}

##📚 Visit http://127.0.0.1:8000/docs for Swagger UI.

---


##⚙️ Setup Instructions
###1. Clone the repository
```bash
git clone https://github.com/yourusername/traffic-sign-classifier.git
cd traffic-sign-classifier```
---


###2. Install dependencies
```bash

pip install -r requirements.txt```

---


###3. Train the model (optional)
```bash
python model_traning.py```

---


###4. Start the FastAPI server
```bash
uvicorn main:app --reload
Make sure model.h5 exists in the root directory.
---
##🧪 Prediction Scripts
For testing without API:

Single Image Prediction: single_img_prediction.py

Batch Prediction: bulk_prediction.py
---
##🛠 Tech Stack
Python 🐍

TensorFlow / Keras

FastAPI

Uvicorn

GTSRB Dataset
---

##📸 Sample Predictions
Images can be placed inside the predicted_classes/ folder and used for testing.
---
##📄 License
Licensed under the MIT License.




