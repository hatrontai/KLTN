import streamlit as st
import numpy as np
import pandas as pd
import os
import json
# import cv2
from PIL import Image
import streamlit.components.v1 as components

# JavaScript để theo dõi phím mũi tên và gửi tín hiệu tới Python
html_code = """
<script>
document.addEventListener('keydown', function(event) {
    let arrowKey = null;
    if (event.key === 'ArrowLeft') {
        arrowKey = 'left';
    } else if (event.key === 'ArrowRight') {
        arrowKey = 'right';
    if (arrowKey) {
        Streamlit.setComponentValue(arrowKey);
    }
});
</script>
"""

st.title("Label font")
data_dir = "D:/Workspace/dataset/Total-Text"
train_image_dir = os.path.join(data_dir, 'train')
image_names = sorted([int(f[3:-4]) for f in os.listdir(train_image_dir) if f.endswith(('jpg', 'jpeg', 'png'))])
# st.write(image_names)
train_label_dir = os.path.join(data_dir, "Annotation/ann_json")
lenght = len(os.listdir(train_image_dir))
font_label = ['serif', 'sans_serif', 'script']
shape_label = ['vertical', 'horizontal', 'curvy']
order_dict = {label: index for index, label in enumerate(font_label)}

if 'img_index' not in st.session_state:
    st.session_state.img_index = 11
# st.write(st.session_state.img_index)

def handle_arrow_key_press(key):
    if key == "left":
        if (st.session_state.img_index-1) in image_names:
            st.session_state.img_index -= 1
            
        else:
            i = st.session_state.img_index -1
            while i not in image_names:
                if i==0:
                    i= image_names[-1]
                    break
                else:
                    i -= 1
            st.session_state.img_index = i

        st.session_state.multiselect = []

    elif key == "right":
        if (st.session_state.img_index+1) in image_names:
            st.session_state.img_index += 1
            
        else:
            i = st.session_state.img_index +1
            while i not in image_names:
                if i > image_names[-1]:
                    i= image_names[0]
                    break
                else:
                    i += 1
            st.session_state.img_index = i

        st.session_state.multiselect = []

def display(index):
    if int(index) not in image_names:
        st.write("none image")
        return
    image_name = f"img{index}.jpg"
    image_path = os.path.join(train_image_dir, image_name)
    label_path = os.path.join(train_label_dir, f"{image_name}.json") 

    with open(label_path, 'r') as openfile:
        # Reading from json file
        ann = json.load(openfile)
    
    image = Image.open(image_path)
    st.image(image, caption= image_name)

    shape_all = st.multiselect("shape for all", shape_label, [], key= "-----")
    col1, spacer, col2, spacer, col6 = st.columns([1, 0.1, 1, 0.1, 1])
    with col1:
        st.title("text")
    with col2: 
        st.title('font')
    with col6:
        st.title('shape')

    for i, object in enumerate(ann['objects']):
        
        col1, spacer, col2, spacer, col6 = st.columns([1, 0.1, 1, 0.1, 1])
        value = object['tags'][0]['value']
        if value == '#':
            continue
        
        with col1:
            st.write("")
            st.write("")
            st.write(value)
        with col2:
            if 'font' in object:
                font = object['font'].split()
                check= 1
                for f in font:
                    if f not in font_label:
                        check= 0
                        break
                if check:
                    font = st.multiselect("", font_label, font, key= i)
                else:
                    font = st.multiselect("", font_label, [], key= i)
            else:
                font = st.multiselect("", font_label, [], key= i)

            sorted_font = sorted(font, key=lambda x: order_dict.get(x, float('inf')))
            object['font'] = ' '.join(sorted_font)

        with col6:
            if 'shape' in object and object['shape'] != []:
                shape = object['shape']
                shape = st.multiselect("", shape_label, shape, key= f"-{i}")
            else:
                if 'font' in object:
                    shape = st.multiselect("", shape_label, shape_all, key= f"-{i}")
                    st.write("none")
                else:
                    shape = st.multiselect("", shape_label, [], key= f"-{i}")
                

            object['shape'] = shape

        points = object['points']['exterior']
        points = np.array(points)
        x_min = min(points[:, 0])
        y_min = min(points[:, 1])
        x_max = max(points[:, 0])
        y_max = max(points[:, 1])

        image_np = np.array(image)
        img = image_np[y_min: y_max, x_min: x_max, :]
        img = Image.fromarray(img)
        st.image(img, caption= '')
            
    if st.button("save", type= "primary"):
        json_object = json.dumps(ann, indent= len(ann))
    
        # Writing to sample.json
        with open(label_path, "w") as outfile:
            outfile.write(json_object)
        
        st.session_state.multiselect = []
        st.write("saved")
    
    # st.write(ann)

col3, col5, col4 = st.columns([0.5, 1, 0.5])

with col3:
    pre = st.button("previous", type= "primary")
    if pre:
        if (st.session_state.img_index-1) in image_names:
            st.session_state.img_index -= 1
            
        else:
            i = st.session_state.img_index -1
            while i not in image_names:
                if i==0:
                    i= image_names[-1]
                    break
                else:
                    i -= 1
            st.session_state.img_index = i

        st.session_state.multiselect = []

with col4:
    next = st.button("next", type= "primary")
    if next:
        if (st.session_state.img_index+1) in image_names:
            st.session_state.img_index += 1
            
        else:
            i = st.session_state.img_index +1
            while i not in image_names:
                if i > image_names[-1]:
                    i= image_names[0]
                    break
                else:
                    i += 1
            st.session_state.img_index = i

        st.session_state.multiselect = []

with col5:
    img_index = int(st.text_input("Index of image", value= st.session_state.img_index, key= "index"))
    st.session_state.img_index = img_index 
        
key_press = components.html(html_code, height=0, width=0)

    # Xử lý tín hiệu từ JavaScript
if key_press:
    handle_arrow_key_press(key_press)

# st.write(st.session_state.img_index)
display(st.session_state.img_index)
