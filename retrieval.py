import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import requests
from PIL import Image
from io import BytesIO

st.title('Image Retrieval')
data_dir = "D:/Workspace/dataset/Total-Text"
train_image_dir = os.path.join(data_dir, 'train')
image_names = sorted([int(f[3:-4]) for f in os.listdir(train_image_dir) if f.endswith(('jpg', 'jpeg', 'png'))])
# st.write(image_names)
train_label_dir = os.path.join(data_dir, "Annotation/ann_json")
lenght = len(os.listdir(train_image_dir))
font_label = ['serif', 'sans_serif', 'script', 'monospaced', 'display']
shape_label = ['vertical', 'horizontal', 'circular', 'curvy']
order_dict = {label: index for index, label in enumerate(font_label)}

font_query = st.multiselect("font query:", font_label, [], key= 'font')
sorted_font = sorted(font_query, key=lambda x: order_dict.get(x, float('inf')))
font_query = ' '.join(sorted_font)

shape_query = st.multiselect('shape query:', shape_label, [], key= 'shape')

col1, spacer, col2 = st.columns([1, 0.2, 1])

def local_css():
    st.markdown(
        """
        <style>
        .fixed-query {
            position: fixed;
            top: 10%;
            left: 5%;
            width: 30%;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Gọi hàm để thêm CSS vào trang
local_css()

with col1:
    query_image_dir = os.path.join(data_dir, "query")
    uploaded_file = st.file_uploader("Load image query", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_query = Image.open(uploaded_file)
        # Mở và hiển thị hình ảnh
        st.markdown('<div class="fixed-query">', unsafe_allow_html=True)
        st.image(image_query, caption="Image query", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        name_image_query = st.text_input('name of image query')
        

        # st.image(image, caption='Image query', use_column_width=True)
with col2:
    if uploaded_file is not None:
        relevant_images = {}
        if font_query != '' and shape_query != []:
            for json_name in os.listdir(train_label_dir):
                json_path = os.path.join(train_label_dir, json_name)
                with open(json_path, 'r') as file:
                    ann = json.load(file)
                
                for object in ann['objects']:
                    if 'font' in object and 'shape' in object:
                        if object['font'] == font_query and object['shape'] == shape_query:
                            image_name = json_name[:-5]
                            image_path = os.path.join(train_image_dir, image_name)
                            image = Image.open(image_path)
                            st.image(image, caption= json_name[:-5])
                            relevant = st.slider('Chọn mức độ relevant:', 0, 5, 0, key= json_name)
                            if relevant > 0:
                                relevant_images[image_name] = relevant
                            break

with col1:
    if uploaded_file is not None:
        relevant_images = dict(sorted(relevant_images.items(), key=lambda item: item[1], reverse= True))
        st.write(relevant_images)
        if name_image_query in os.listdir(query_image_dir):
            st.write("name already exists, please anter another name")
        else:
            save = st.button("save", type= "primary")

            image_query_path = os.path.join(query_image_dir, f"{name_image_query}.png")
            if save:
                image_query.save(image_query_path, format="PNG")

                label_path = os.path.join(query_image_dir, 'relevant.json')
                with open(label_path, 'r') as file:
                    label = json.load(file)
                label[name_image_query] = relevant_images

                json_label = json.dumps(label, indent= len(label))
                with open(label_path, "w") as outfile:
                    outfile.write(json_label)

                st.write('saved')
    else:
        st.write("please upload image query")

