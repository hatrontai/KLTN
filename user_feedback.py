import os
import json
import streamlit as st
from PIL import Image
import glob

# Function to load results from the provided JSON file (fix for Streamlit UploadedFile)
def load_results(uploaded_json_file):
    # Read the contents of the uploaded JSON file
    file_contents = uploaded_json_file.read().decode('utf-8')
    results = json.loads(file_contents)
    return results

# Function to save labels immediately to a JSON file
def save_labels_immediately(labeled_data, labels_path):
    with open(labels_path, 'w') as f:
        json.dump(labeled_data, f, indent=4)
    st.success("Labels saved successfully!")

# Streamlit app interface
def streamlit_app():
    # Title of the app
    st.title('Image Retrieval Evaluation')

    # Step 1: Input the folder containing query images
    query_folder = st.sidebar.text_input("Enter the folder path for query images", 'D:/Workspace/Project/KLTN/query')
    query_images = glob.glob(os.path.join(query_folder, "*.jpg"))
    
    if len(query_images) == 0:
        st.error("No query images found in the specified folder.")
        return

    # Step 2: Input the folder containing result images
    result_folder = st.sidebar.text_input("Enter the folder path for result images", 'D:/Workspace/Project/KLTN/database/data')
    result_images = glob.glob(os.path.join(result_folder, "*.jpg"))
    
    if len(result_images) == 0:
        st.error("No result images found in the specified folder.")
        return

    # Step 3: Input the JSON file containing the retrieval results
    json_file = st.sidebar.file_uploader("Upload the JSON file with retrieval results", type=["json"])
    if json_file is not None:
        results = load_results(json_file)
    else:
        st.error("Please upload a JSON file with retrieval results.")
        return

    labels_path = os.path.join('D:/Workspace/Project/KLTN/labeled_results', f'label_{json_file.name}')
    # Step 4: Create a folder to store labeled data if it doesn't exist
    if not os.path.exists(labels_path):
        # os.makedirs(labels_path)
        labeled_data = {}
    else:    
        # st.write('ssvakjvdnsakj')
        with open(labels_path, 'r') as file:
            labeled_data = json.load(file)

    # Step 5: Display the results for each query image
    # for query_image in query_images:  # Modify this to handle all query images if needed
    query_names = [os.path.basename(query_image) for query_image in query_images]
    query_name =  st.sidebar.selectbox('Load image', query_names)
    query_image = os.path.join(query_folder, query_name)

    list_image_non_label = [q for q in query_names if q not in list(labeled_data.keys())]
    # Step 6: Add scrollable container for both query image and results
    with st.container():
        # Query Image displayed on top
        st.sidebar.image(query_image, caption=f"Query Image: {query_name}", use_column_width=True)
        st.sidebar.write(list_image_non_label)

        # Retrieve the top 30 results for this query
        top_30_results = results.get(query_name, [])
        # if not top_30_results:
        #     st.write("No results found for this query.")
        #     continue

        # Create a scrollable section for the results
        st.write("**Results**")
        st.write("Scroll through the results and label them.")

        # Scrollable layout for results (along with query image)
        result_container = st.container()
        with result_container:
            labels = labeled_data.get(query_name, {})
            for idx, result_image in enumerate(top_30_results):  # Limit to 5 results for quicker testing
                with st.container():
                    # col1, col2 = st.columns(2)
                    # with col1:
                        result_image_path = os.path.join(result_folder, result_image + ".jpg")
                        if os.path.exists(result_image_path):
                            img = Image.open(result_image_path)
                            st.image(img, caption=f"Result {idx+1}: {result_image}", use_column_width=False)
                            
                            # Label the result
                            label = st.radio(f"Is this result relevant for query {query_name}?", options=["relevant", "not relevant"], index= labels.get(result_image, 1), key=f"{query_name}_{idx}")
                            labels[result_image] = 0 if label == 'relevant' else 1 
                            
                            # Immediately save the label after the annotator selects it
                            labeled_data[query_name] = labels
                            # save_labels_immediately(labeled_data, labels_path)  # Save after every label change
                        else:
                            st.write(f"Result image {result_image} not found in the result folder.")
                    # with col2:
                    #     if os.path.isfile(os.path.join('D:/Workspace/dataset/Total-Text/Train', f"{result_image.split('.')[0]}.jpg")):
                    #         or_image_path = os.path.join('D:/Workspace/dataset/Total-Text/Train', f"{result_image.split('.')[0]}.jpg")
                    #     else:
                    #         or_image_path = os.path.join('D:/Workspace/dataset/Total-Text/Train', f"{result_image.split('.')[0]}.JPG")

                    #     or_image = Image.open(or_image_path)
                    #     st.image(or_image, caption= result_image.split('.')[0], use_column_width= True)
        # Store the labels for the current query (optional, as it's already being saved immediately)
        labeled_data[query_name] = labels

    # Save the labeled data after all results have been labeled (this will not be necessary if saving immediately after each label)
    if st.button("Save Labels"):
        save_labels_immediately(labeled_data, labels_path)
    

# Run the app
if __name__ == "__main__":
    streamlit_app()

# color 1553