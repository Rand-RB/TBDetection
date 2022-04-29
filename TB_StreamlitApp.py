import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

st.write("""
# Chest X_Ray  **TB Prediction** type!
""")

model = tf.keras.models.load_model("chest_xray_model.h5")
uploaded_img = st.file_uploader("Choose an Chest X-Ray Image the extention should be '.jpg'")

if  uploaded_img is not None:

    img_bytes = np.asarray(bytearray(uploaded_img.read()), dtype = np.uint8) # Convert to an opencv image.
    cv_Img = cv2.imdecode(file_bytes, 1)
    cv_Img  = cv2.cvtColor(cv_Img, cv2.COLOR_BGR@RGB)
    img_eqhist=cv2.equalizeHist(cv_Img)
    clahe = cv2.createCLAHE(clipLimit=3).apply(img_eqhist)
    invert = cv2.bitwise_not(clahe)
    resize = cv2.resize(opencv_image,(224,224))
    st.image(opencv_image)

pred = st.button("Let's See The  Tuberculosis Prediction Result ")

if pred:
    my_pred = model.predict(resize)
    result = int(my_pred [0][0])
    if (result == 0):
        st.title("Patient is Affected By Tuberculosis")
    else:
        st.title("The patient's Chest is Normal")



