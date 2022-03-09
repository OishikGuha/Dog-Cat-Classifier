import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

MODEL = tf.keras.models.load_model("v2.h5")
CLASS_NAMES = ["Cat", "Dog"]

st.write("# Dog vs cat classifier")

image = st.file_uploader("Upload image", type=[
                         ".jpg", ".png", ".jpeg", ".webp"])


if image:
    st.image(image)


if st.button("Predict!") and image is not None:
    image = Image.open(image)
    image = image.resize((256, 256))
    imageArray = np.expand_dims(np.array(image), 0)
    predictions = MODEL.predict(imageArray)

    st.subheader("The Ai thinks that this is a " +
                 CLASS_NAMES[np.argmax(predictions)])
else:
    st.write("Please input an image!")
