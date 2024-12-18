import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import base64

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #748A88;
    }
</style>
""", unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('Thiet-ke-shop-giay-1.png')  

#st.sidebar.header("Nhận dạng trái cây")

# Load YOLOv8 model
if 'model' not in st.session_state:
    st.session_state.model = YOLO('best.pt')  # Replace 'best.pt' with your YOLOv8 model path

st.subheader('Nhận dạng giày chính hãng')

FRAME_WINDOW = st.image([])

# Default image
image = cv2.imread('NoImage.bmp', cv2.IMREAD_COLOR)
FRAME_WINDOW.image(image, channels='BGR')

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["bmp", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    FRAME_WINDOW.image(opencv_image, channels='BGR')
    press = st.button('Nhận dạng')
    if press:
        # Run YOLOv8 model inference
        st.write("Đang nhận dạng...")
        results = st.session_state.model(opencv_image)

        # Extract predictions
        predictions = results[0]
        boxes = predictions.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = predictions.boxes.conf.cpu().numpy()  # Confidence scores
        classes = predictions.boxes.cls.cpu().numpy().astype(int)  # Class indices
        class_names = st.session_state.model.names  # Class names from YOLOv8 model

        # Draw bounding boxes
        for i in range(len(boxes)):
            if scores[i] >= 0.7:  # Only show detections with confidence >= 0.7
                x1, y1, x2, y2 = map(int, boxes[i])
                label = f"{class_names[classes[i]]} ({scores[i]:.2f})"
                color = (0, 255, 0)  # Green for bounding box
                opencv_image = cv2.rectangle(opencv_image, (x1, y1), (x2, y2), color, 2)
                opencv_image = cv2.putText(opencv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display results
        FRAME_WINDOW.image(opencv_image, channels='BGR')
