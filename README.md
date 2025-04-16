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
dataset/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── validation/
    ├── class_0/
    ├── class_1/
    └── ...


