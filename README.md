# YOLO-Image-Component-Detection
**Link:** https://yolo-image-component-detection.streamlit.app/
## Overview

This application allows you to upload an image and analyze it using various YOLO models to detect different components within the image. The results include bounding boxes and class names for the detected objects.

## Features

- **Upload Image**: Upload an image in JPG, JPEG, or PNG format.
- **Select Model**: Choose from multiple YOLO models (`yolov5` and `yolov8` series) for analysis.
- **Set Confidence Threshold**: Adjust the confidence threshold for object detection.
- **View Results**: Display the original image, detected components, and image with bounding boxes.

## How to Use

1. **Upload an Image**: Click on the "Choose an image..." button to upload your image.
2. **Select Model**: Use the sidebar to choose the YOLO model for detection.
3. **Set Confidence Threshold**: Adjust the confidence threshold slider in the sidebar to refine detection sensitivity.
4. **Analyze Image**: Click on the "Analyse Image" button to start the detection process.
5. **View Results**: The detected components and the image with bounding boxes will be displayed.

## Dependencies

- `streamlit`
- `PIL` (Pillow)
- `ultralytics`
- `opencv-python`
- `numpy`

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone (https://github.com/faridaelhusseiny278/YOLO-Image-Component-Detection.git)
   cd YOLO-Image-Component-Detection
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
