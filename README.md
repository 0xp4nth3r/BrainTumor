#  Brain Tumor Detection and Segmentation with YOLOv8


An AI-based tool for detecting and segmenting brain tumors from CT and X-ray images using YOLOv8, deployed with a user-friendly Streamlit interface.

---

##  Project Overview

This project allows healthcare professionals or researchers to upload medical images (CT, MRI, or X-rays) and receive real-time segmentation results for brain tumors. It uses the **YOLOv8 segmentation model** to detect and outline tumor regions, and it presents the results in an intuitive web app built with **Streamlit**.

>  Designed for real-time use  
>  Built with a focus on interpretability  
>  Useful as a reference for AI agent logic in security or diagnostic systems

---

##  Features

-  **YOLOv8 Segmentation**: Detects and segments tumors in real-time
-  **Image Upload & Preview**: Drag-and-drop CT/X-ray scans
-  **Confidence Thresholding**: Filters out low-probability detections
-  **Medical Guidance**: Shows causes, symptoms, and precautions if a tumor is detected
-  **Colormap Generator**: Applies 12 OpenCV color maps to visualize image features
-  **Lightweight Deployment**: Built entirely in Python using Streamlit

---

##  Example Workflow

1. Upload a CT or X-ray image  
2. Submit for analysis  
3. View:
   - Original scan
   - YOLOv8-segmented result
   - Medical context and suggestions
4. Generate and view color-mapped variants of the scan

---

##  Tech Stack

| Component     | Purpose                             |
|--------------|-------------------------------------|
| `ultralytics`| YOLOv8 model inference              |
| `Streamlit`  | Web-based frontend interface        |
| `OpenCV`     | Image colormap transformations      |
| `PIL`        | Image handling and conversion       |

---

##  Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/0xp4nth3r/BrainTumor.git
   cd BrainTumor
