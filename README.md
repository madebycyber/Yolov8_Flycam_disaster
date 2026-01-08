# Project: Disaster Victim Detection using Unity Simulation \& YOLOv8

## 📖 Introduction

This project develops a simulated Unmanned Aerial Vehicle (UAV) system in **Unity** combined with **YOLOv8** AI to detect and locate victims in post-disaster scenarios (floods, landslides, forest fires) in Vietnam.

\[cite\_start]The system follows a closed-loop pipeline: **Unity Simulation → Data Generation → AI Training → 3D Localization**, aiming to support search and rescue operations safely and effectively\[cite: 1, 8, 20].

---

## 🛠️ System Requirements

### Hardware

* \[cite\_start]**CPU**: Intel Core i7 or equivalent (Tested on i7, 8GB RAM)\[cite: 90].
* \[cite\_start]**GPU**: NVIDIA GPU recommended for faster training (CUDA support)\[cite: 32].

### Software \& Libraries

* \[cite\_start]**Unity (2022 LTS)**: For 3D environment simulation\[cite: 45].
* **Python**: Primary programming language.
* \[cite\_start]**Ultralytics YOLOv8**: For object detection model training\[cite: 45].
* \[cite\_start]**Roboflow**: For data annotation and management\[cite: 45].
* **Others**: `OpenCV`, `NumPy`, `Matplotlib`, `Jupyter Notebook`.

---

## 🚀 Methodology Pipeline

### Phase 1: Simulation Data Generation (Unity)

\[cite\_start]We use Unity to recreate disaster environments and collect synthetic training data\[cite: 23].

* **Environments**: Three terrain types were built: **Mountainous, Rural, and Urban**.
* **Scenarios**: Floods, Landslides, Forest fires.
* \[cite\_start]**Conditions**: Day/Night cycles, Fog, Rain effects\[cite: 23].
* **Data Collection Process**:

  * \[cite\_start]**UAV Logic**: Scripted in C# to fly automatically (straight or circular paths) at altitudes of **50–80m** covering >80% of the area\[cite: 39, 51].
  * \[cite\_start]**Camera**: Captures images at 1920x1080 (resized to 640x640 for training) with angles between 0–60°\[cite: 39, 50].
  * \[cite\_start]**Domain Randomization (DR)**: To improve model robustness, we randomized sky backgrounds, lighting, object colors, and weather conditions\[cite: 25, 122].

### Phase 2: Data Preparation (Roboflow)

\[cite\_start]Images collected from Unity are processed and labeled using Roboflow\[cite: 55].

* \[cite\_start]**Labeling**: Manual bounding box annotation for the "Victim" class\[cite: 56].
* **Augmentation**: Applied to increase diversity, including:

  * Rotation \& Flipping.
  * Brightness/Contrast adjustments.
  * Noise addition.

* \[cite\_start]**Dataset Split**\[cite: 57, 58]:

  * **Train**: ~951 images.
  * **Valid**: ~260 images.
  * **Test**: ~219 images (Includes 50 real-world "out-of-domain" images with blurred faces for privacy and realistic testing).

### Phase 3: Training \& Fine-tuning

\[cite\_start]We utilized **YOLOv8 Nano (n)** for its balance between speed and accuracy, suitable for personal computing hardware\[cite: 29].

#### \[cite\_start]Hyperparameters\[cite: 61]:

* **Model**: `yolov8n.pt` (Pre-trained)
* **Epochs**: 100
* **Batch size**: 16
* **Optimizer**: Adam
* **Image Size**: 640
* **Device**: GPU (recommended) or CPU

#### Training Command (Example)

```python
from ultralytics import YOLO

# 1. Load the pre-trained model
model = YOLO('yolov8n.pt') 

# 2. Train the model on the custom dataset
results = model.train(
    data='path/to/data.yaml',  # Exported from Roboflow
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='Adam',
    name='victim\_detection\_v1'
)

