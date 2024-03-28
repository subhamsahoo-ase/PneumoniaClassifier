import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):

    """
    This function sets background as bimage file

    Parameters:
    image_file : str : image file path

    Returns:
    None
    """

    with open(image_file, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    style = f"""
    <style>
    .stApp{{
        background-image: url('data:image/png;base64,{base64_image}');
        background-size: cover;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    This function classifies the image

    Parameters:
    image : PIL.Image : image to classify
    model : keras.models.Model : model to use for classification
    class_names : list : list of class names

    Returns:
    Tuple: (str, float) : class name and confidence
    """
    # CONVERT IMAGE TO (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # CONVERTS IMAGE TO NUMPY ARRAY
    image_array = np.asarray(image)

    # NORMALIZE IMAGE
    nomralized_image_array = (image_array.astype(np.float32) / 127.) - 1

    # SET MODEL INPUT
    data = np.ndarray(shape = (1, 224, 224, 3), dtype=np.float32)
    data[0] = nomralized_image_array

    # PREDICT
    prediction = model.predict(data)

    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    condfidence_score = prediction[0][index]

    return class_name, condfidence_score
