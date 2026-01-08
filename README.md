# Project: Disaster Victim Detection using Unity Simulation \& YOLOv8

### Introduction

This project develops a simulated Unmanned Aerial Vehicle (UAV) system in **Unity** combined with **YOLOv8** AI to detect and locate victims in post-disaster scenarios (floods, landslides, forest fires) in Vietnam.

The system follows a closed-loop pipeline: **Unity Simulation → Data Generation → AI Training → 3D Localization**, aiming to support search and rescue operations safely and effectively.

---

### System Requirements

#### Hardware

* **CPU**: Intel Core i7 or equivalent (Tested on i7, 8GB RAM).
* **GPU**: NVIDIA GPU recommended for faster training (CUDA support).

#### Software \& Libraries

* **Unity (2022 LTS)**: For 3D environment simulation.
* **Python**: Primary programming language.
* **Ultralytics YOLOv8**: For object detection model training.
* **Roboflow**: For data annotation and management.
* **Others**: `OpenCV`, `NumPy`, `Matplotlib`, `Jupyter Notebook`.

---

### **Methodology Pipeline**

#### Phase 1: Simulation Data Generation (Unity)

We use Unity to recreate disaster environments and collect synthetic training data.

* **Environments**: Three terrain types were built: **Mountainous, Rural, and Urban**.
* **Scenarios**: Floods, Landslides, Forest fires.
* **Conditions**: Day/Night cycles, Fog, Rain effects\[cite: 23].
* **Data Collection Process**:

  * **UAV Logic**: Scripted in C# to fly automatically (straight or circular paths) at altitudes of **50–80m** covering >80% of the area.
  * **Camera**: Captures images at 1920x1080 (resized to 640x640 for training) with angles between 0–60°.
  * **Domain Randomization (DR)**: To improve model robustness, we randomized sky backgrounds, lighting, object colors, and weather conditions.

#### Phase 2: Data Preparation (Roboflow)

Images collected from Unity are processed and labeled using Roboflow.

* **Labeling**: Manual bounding box annotation for the "Victim" class.
* **Augmentation**: Applied to increase diversity, including:

  * Rotation \& Flipping.
  * Brightness/Contrast adjustments.
  * Noise addition.

* **Dataset Split**\[cite: 57, 58]:

  * **Train**: ~951 images.
  * **Valid**: ~260 images.
  * **Test**: ~219 images (Includes 50 real-world "out-of-domain" images with blurred faces for privacy and realistic testing).

#### Phase 3: Training \& Fine-tuning

We utilized **YOLOv8 Nano (n)** for its balance between speed and accuracy, suitable for personal computing hardware.

##### Hyperparameters:

* **Model**: `yolov8n.pt` (Pre-trained)
* **Epochs**: 100
* **Batch size**: 16
* **Optimizer**: Adam
* **Image Size**: 640
* **Device**: GPU (recommended) or CPU

##### Training Command (Example)

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
    name='victim\\\\\\\_detection\\\\\\\_v1'
)



