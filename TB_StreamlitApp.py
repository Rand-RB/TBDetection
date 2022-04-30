import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import pickle
import tensorflow as tf
import cv2
#from .cv2 import *

st.write("""
# Chest X_Ray  **TB Prediction** type!
""")

model = tf.keras.models.load_model("chest_xray_model.h5.pkl")
uploaded_img = st.file_uploader("Choose an Chest X-Ray Image the extention should be '.jpg'")

if  uploaded_img is not None:

    img_bytes = np.asarray(bytearray(uploaded_img.read()), dtype = np.uint8) # Convert to an opencv image.
    cv_Img = cv2.imdecode(img_bytes, 1)
    #img = cv2.imread(cv_Img, 0)
    gray_img = cv2.cvtColor(cv_Img,cv2.COLOR_BGR2GRAY)
    img_hist =cv2.equalizeHist(gray_img)
    clahe = cv2.createCLAHE(clipLimit=3).apply(img_hist)
    invert = cv2.bitwise_not(clahe)
    resized_img = cv2.resize(invert,[512,512],3)
    st.image(resized_img)

pred = st.button("Let's See The  Tuberculosis Prediction Result ")

if pred:
    my_pred = model.predict(resized_img, shape = [512,512,3])
    result = int(my_pred [0][0])
    if (result == 0):
        st.title("Patient is Affected By Tuberculosis")
    else:
        st.title("The patient's Chest is Normal")




