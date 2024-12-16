import streamlit as st
import numpy as np
import pandas as pd
import os
import json
# import cv2
from PIL import Image
import pandas as pd
from Project.KLTN.util.infer import get_font_embedding

# Screen streamlit
st.set_page_config(layout="wide")

st.title("DEMO APP RETRIEVAL IMAGE")
# st.markdown("")

# Input Parameters - Sidebar
st.sidebar.title("Input Parameters")

st.sidebar.markdown('### Load input: image and text query')
# load image query
uploaded_file = st.sidebar.file_uploader("Load image query", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image_query = Image.open(uploaded_file)
    # Mở và hiển thị hình ảnh
    st.sidebar.image(image_query, caption="Image query", use_column_width=True)
    # name_image_query = st.sidebar.text_input('name of image query', value= uploaded_file.name.split('.')[0])
    # st.sidebar.write(name_image_query)

# text query
text_query = st.sidebar.text_input(label= 'Text query', key= 'text_query')

# feature check
st.sidebar.markdown("### Retrieval with feature:")
font = st.sidebar.checkbox(label= "Font", value= True, key= 'font_check')
shape = st.sidebar.checkbox(label= "Shape", value= True, key= 'shape_check')
color = st.sidebar.checkbox(label= "Color", value= True, key= 'color_check')



# Output retrieved - main screen
st.write(get_font_embedding())