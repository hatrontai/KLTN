import matplotlib.pyplot as plt
from PIL import Image
from util.embedding import get_color_embedding, get_font_embedding, get_shape_embedding, load_style_embedding
import faiss
import numpy as np
import json
import os
import glob
from Levenshtein import distance
from util.focus_text import focus_text, ocr_model


img_list = glob.glob('./database/image/*.jpg')
file_path = './database/ann.json'
with open(file_path, 'r') as json_file:
    ann = json.load(json_file)

text_list = []
for pth in img_list:
    img_name = os.path.basename(pth)
    value = ann.get(img_name,{}).get('value')
    if value is not None:
        text_list.append(ann.get(img_name,{}).get('value'))
    else:
        text_list.append("")

style_embeddings, mapping = load_style_embedding()

def calculate_final_scores(faiss_results, faiss_distances, levenshtein_results, levenshtein_distances, alpha=0.5, beta=0.5):
    """
    Sửa cái hàm này thành get combine score of 2 feature
    Calculate final scores by combining FAISS and Levenshtein results.

    Args:
        faiss_results (list): List of image names from FAISS search.
        faiss_distances (list): Corresponding FAISS distances or similarity scores.
        levenshtein_results (list): List of image names from Levenshtein search.
        levenshtein_distances (list): Corresponding Levenshtein distances.
        alpha (float): Weight for FAISS scores.
        beta (float): Weight for Levenshtein scores.

    Returns:
        list: Sorted list of tuples (image_name, final_score).
    """
   
    faiss_scores = np.max(faiss_distances)  - faiss_distances  
    faiss_scores = faiss_scores / np.max(faiss_scores)
    # Normalize Levenshtein distances (lower distance is better, so invert)
    # Đối với feature index = faiss thì làm y như trên, chỉ có text là tính score như bên dưới
    levenshtein_scores = np.max(levenshtein_distances) - np.array(levenshtein_distances)
    levenshtein_scores = levenshtein_scores / np.max(levenshtein_scores)
    
    score_dict = {}
    
    # Add FAISS scores
    for i, img_name in enumerate(faiss_results):
        if img_name not in score_dict:
            score_dict[img_name] = {'faiss_score': faiss_scores[i], 'levenshtein_score': 0}
        else:
            score_dict[img_name]['faiss_score'] = faiss_scores[i]
    
    # Add Levenshtein scores
    for i, img_name in enumerate(levenshtein_results):
        if img_name not in score_dict:
            score_dict[img_name] = {'faiss_score': 0, 'levenshtein_score': levenshtein_scores[i]}
        else:
            score_dict[img_name]['levenshtein_score'] = levenshtein_scores[i]
   
    final_scores = []
    for img_name, scores in score_dict.items():
        # print(scores)
       
        final_score = (scores['faiss_score'] + 0 *scores['levenshtein_score'])
        final_scores.append((img_name, final_score))
    
    # Sort by final score in descending order
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores
 
def get_text_relevant_list(text_query, text_list):
    """
    Calculate Levenshtein distances for a query text and return sorted texts with distances.

    Args:
        text_query (str): The query text.
        text_list (list): List of texts to compare against the query.

    Returns:
        tuple: (sorted_texts, sorted_distances)
    """
    # Calculate Levenshtein distances
    if text_query is None or text_query == '':
        dis_list = [0 for text in text_list]
    else:
        dis_list = [distance(text_query.upper(), text.upper())/len(text_query) for text in text_list]
    
    # Convert to a numpy array for efficient sorting
    dis_array = np.array(dis_list)
    
    # Get sorted indices
    sorted_indices = np.argsort(dis_array)
    
    # Create sorted lists
    sorted_distances = dis_array[sorted_indices]
    sorted_text = [text_list[id] for id in sorted_indices]
    # print(sorted_text[:10])
    # _, mapping = load_style_embedding()
    # temp = [mapping[id] for id in sorted_indices]
   
    return  sorted_distances,sorted_indices

def inference(text_query = None, image_query_path = None, rm_background = True, feature_list = ['font','shape','color']):
    """ 
    Get result list for a query
    Args:
        text_query (str): The query text
        image_query (PIL image): The query image
        feature_list (list of text): The feature want to infer
        ['text','font','shape','color','all']
    Returns:
        tuple: (sorted_index, sorted_score)
    """

    if rm_background:
        image = focus_text(image_query_path, ocr_model)
    else:
        image = Image.open(image_query_path)

    # plt.imshow(image)
    # plt.show()
    
    font = get_font_embedding(image)
    shape = get_shape_embedding(image)
    color = get_color_embedding(image)
    
    if 'font' in feature_list:
        font =  font/len(font)
        font = font.numpy().reshape(1,-1).astype('float32')
        faiss.normalize_L2(font)
    else:
        font = np.full(font.shape, None)
        
    if 'shape' in feature_list:    
        shape =  shape/len(shape)
        shape = shape.numpy().reshape(1,-1).astype('float32')
        faiss.normalize_L2(shape)
    else:
        shape = np.full(shape.shape, None)
        
    if 'color' in feature_list:
        color =  color/len(color)
        color = color.reshape(1,-1).astype('float32')
        faiss.normalize_L2(color)
    else:
        color = np.array([np.full(color.shape, None)])
        
    # print('font embedding:', font.shape)
    # print('shape embedding:', shape.shape)
    # print('color embedding:', color)

    if 'font' in feature_list:
        concat = np.concatenate([font*3, color, shape],axis = 1)
    else:
        concat = np.concatenate([font, color, shape],axis = 1)

    # style_embeddings,_ = load_style_embedding()
    style_embed = []
    query_embed = np.array(concat)
    # print(style_embeddings.shape, query_embed.shape)
    for embed in np.array(style_embeddings):
        new_embed = embed[query_embed[0] != None]
        style_embed.append(new_embed)
    style_embed = np.array(style_embed).astype('float32')
    # print(style_embed.shape)

    index = faiss.IndexFlatL2(style_embed.shape[1])
    index.add(style_embed)

    query_embed = query_embed[query_embed != None]
    query_embed = np.array([query_embed])
    query_embed = query_embed.astype('float32')
    # print(query_embed)
    faiss.normalize_L2(query_embed)
    # return concat
    # print(concat)
    style_distances ,style_indices = index.search(np.array(query_embed),k=9000)
    # color_distances , color_indices = color_index.search(np.array(color_embedding),k=9999)
    # shape_distances, shape_indices = shape_index.search(np.array(shape_embedding,k=9999))
    # font_distances,font_indices = font_index.search(np.array(font_embedding,k=99999))
    text_distances, text_indices = get_text_relevant_list(text_query,text_list)
    final_list = calculate_final_scores(style_indices[0],style_distances[0],text_indices,text_distances)
  
    retrieved_images = []
   
    for _,res in enumerate(final_list):
     
        if(res[0] not in mapping):
            continue
        img_name = mapping[res[0]].split('.')[0]
        if(img_name not in retrieved_images):
            retrieved_images.append(img_name)
   
    results = [mapping[id[0]] for _,id in enumerate(final_list) if id[0] in mapping ]
    return image, results
 
    
def visualize_results(results, base_path):
    # Limit to the first 5 results
    top_results = results[:20]
    
    plt.figure(figsize=(20, 10))  # Set the figure size for better visibility
    
    for i, result in enumerate(top_results):
        image_path = f"{base_path}/{result}.jpg"  # Create the image path
        img = Image.open(image_path)  # Open the image
        
        # Plot the image
        plt.subplot(2, 10, i + 1)  # Create a subplot for each image (1 row, 5 columns)
        plt.imshow(img)
        plt.title(f"Result {i+1}")
        plt.axis('off')  # Hide axis

    plt.show()