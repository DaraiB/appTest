import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

model= keras.models.load_model('./model.h5') свою модель выгрузить и свой путь прописать

st.title('Определение вида птицы')


uploaded_image = st.file_uploader("Загрузите изображение птицы", type=["jpg", "png", "jpeg"])


if uploaded_image is not None:  

    img = image.load_img(uploaded_image, target_size=(224, 224), grayscale=False)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)


    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    max_probability = np.max(predictions)
    

    st.image(uploaded_image, caption='Загруженное изображение', use_column_width=True)
    st.write(f'Максимальная вероятность: {max_probability:.2%}')
