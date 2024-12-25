import streamlit as st
import numpy as np
import pandas as pd
import os
import json
# import cv2
from PIL import Image
import pandas as pd
from util.infer import inference
import matplotlib.pyplot as plt


test_dir = './image_test'
base_path = './database/image'
Total_text_path = 'D:/Workspace/dataset/Total-Text/Train'
base_ann_path = './database/ann.json'

with open(base_ann_path, 'r') as file:
    anns = json.load(file)


# Screen streamlit
st.set_page_config(layout="wide")

st.title("DEMO APP RETRIEVAL IMAGE")
# st.markdown("")

# Input Parameters - Sidebar
st.sidebar.title("Input Parameters")

st.sidebar.markdown('### Load input: image and text query')

# feature check
# st.sidebar.markdown("### Retrieval with feature:")
feature_list = ['font', 'shape', 'color']
feature_list = st.sidebar.multiselect("Retrieval with feature:", feature_list, feature_list, key= 'choose_feature')

# text query
text_query = st.sidebar.text_input(label= 'Text query', key= 'text_query')
# load image query
uploaded_file = st.sidebar.file_uploader("Load image query", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    if text_query is None:
        st.write("Please input text query")
        
    image_query = Image.open(uploaded_file)
    # Mở và hiển thị hình ảnh

    st.sidebar.image(image_query, caption="Image query", use_column_width=True)
    image_query_path = os.path.join(test_dir, uploaded_file.name)
    image_query.save(image_query_path, format="PNG")
    # name_image_query = st.sidebar.text_input('name of image query', value= uploaded_file.name.split('.')[0])
    # st.sidebar.write(name_image_query)

    # Output retrieved - main screen
    image, results = inference(text_query= text_query, image_query_path= image_query_path, rm_background= False, feature_list=feature_list)
    # st.image(image, caption= "Image query after processing")
    # col1, spacer, col2 = st.columns([0.6, 0.05, 1])
    for i, img_name in enumerate(results):
        if i > 40:
            break
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                img_path = f"{base_path}/{img_name}.jpg"  # Create the image path
                img = Image.open(img_path)

                st.image(img, caption=f"Results {i}")
                # st.write(anns[f'{img_name}.jpg'])

            with col2:
                if os.path.isfile(os.path.join(Total_text_path, f"{img_name.split('.')[0]}.jpg")):
                    image_path = os.path.join(Total_text_path, f"{img_name.split('.')[0]}.jpg")
                else:
                    image_path = os.path.join(Total_text_path, f"{img_name.split('.')[0]}.JPG")

                image = Image.open(image_path)
                st.image(image, caption= img_name.split('.')[0], use_column_width= True)
