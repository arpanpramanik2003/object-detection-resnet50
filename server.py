import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("cifar10_resnet50_model.h5")

labels_dictionary = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 
                     4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("CIFAR-10 Image Classification using ResNet50")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction) 
    predicted_class = labels_dictionary[predicted_index]

    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", f"{np.max(prediction) * 100:.2f}%")

    prediction_dict = {labels_dictionary[i]: prediction[0][i] for i in range(10)}
    st.bar_chart(prediction_dict)
