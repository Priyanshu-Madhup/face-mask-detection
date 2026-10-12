import os

# TensorFlow 2.16+ ships Keras 3 by default, which cannot deserialize this
# repo's model (saved with old-style Keras 2 configs). Force the legacy
# Keras 2 compatibility shim (tf-keras) before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_resource
def load_face_detector():
    prototxt_path = os.path.join("face_detector", "deploy.prototxt")
    weights_path = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    return cv2.dnn.readNet(prototxt_path, weights_path)


@st.cache_resource
def load_mask_model():
    return load_model("mask_detector.model")


def mask_image(image_bgr):
    """Run face detection + mask classification on a BGR image (numpy array).
    Returns an RGB image with annotations and the number of faces found."""
    net = load_face_detector()
    model = load_mask_model()

    image = image_bgr.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces_found = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue
            faces_found += 1

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face, verbose=0)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img, faces_found


def mask_detection():
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Webcam"]
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            our_image = Image.open(image_file).convert("RGB")
            st.image(our_image, caption='', use_container_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                with st.spinner("Detecting faces and masks..."):
                    image_bgr = cv2.cvtColor(np.array(our_image), cv2.COLOR_RGB2BGR)
                    rgb_img, faces_found = mask_image(image_bgr)
                if faces_found == 0:
                    st.warning("No faces detected in the image.")
                else:
                    st.success(f"Detected {faces_found} face(s).")
                st.image(rgb_img, use_container_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        picture = st.camera_input("Take a picture")
        if picture is not None:
            our_image = Image.open(picture).convert("RGB")
            with st.spinner("Detecting faces and masks..."):
                image_bgr = cv2.cvtColor(np.array(our_image), cv2.COLOR_RGB2BGR)
                rgb_img, faces_found = mask_image(image_bgr)
            if faces_found == 0:
                st.warning("No faces detected in the image.")
            else:
                st.success(f"Detected {faces_found} face(s).")
            st.image(rgb_img, use_container_width=True)


mask_detection()
