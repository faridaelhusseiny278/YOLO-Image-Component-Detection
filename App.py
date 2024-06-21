import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np

def load_model(model_name):
    return YOLO(f"{model_name}.pt")

def predict(model, img, conf=0.5):
    results = model.predict(img, conf=conf)
    class_names = [model.names[int(box.cls)] for result in results for box in result.boxes]
    class_counts = Counter(class_names)
    formatted_output = ", ".join([f"{count} {cls}" + ("s" if count > 1 else "") for cls, count in class_counts.items()])
    return formatted_output

def predict_and_detect(model, img, conf=0.5):
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv2, conf=conf)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_cv2, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return img_rgb, results

st.title("🔍 Image Component Detection using YOLO")
st.write("Upload an image and click 'Analyse Image' to see the detected components.")

st.sidebar.title("⚙️ Configuration")
confidence = st.sidebar.slider('Confidence Threshold', 0.1, 1.0, 0.5)
st.sidebar.write(f"Current confidence threshold: {confidence}")

model_names = [ 'yolov5su', 'yolov5m', 'yolov5l', 'yolov5x',
                'yolov8n','yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
selected_model_name = st.sidebar.selectbox('Select YOLO model', model_names)

model = load_model(selected_model_name)

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Uploaded Image', use_column_width=False, width=700)
    st.write("")
    
    if st.button('🔍 Analyse Image'):
        st.write("🔄 Classifying...")
        
        with st.spinner('Processing...'):
            detected_objects = predict(model, image)
            img, _ = predict_and_detect(model, image)
        
        st.success("✅ Analysis Complete")
        
        st.subheader("Detected Components:")
        st.write(detected_objects)
        
        st.subheader("Image with Detected Components:")
        st.image(img, caption='📸 Detected Image', use_column_width=False, width=700)

st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
        }
        .css-1aumxhk {
            margin-bottom: 20px;
        }
        .css-1aumxhk img {
            border: 2px solid #004aad;
            border-radius: 10px;
        }
        .css-1aumxhk h2, .css-1aumxhk h3 {
            color: #004aad;
        }
    </style>
    """, unsafe_allow_html=True)
