from glob import glob
import os
import faiss
import json
from Levenshtein import distance
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from PIL import Image 
from collections import Counter
from tqdm.notebook import tqdm
import time
import cv2
import torch.nn.functional as F
from util import canvas as cav
import torch.backends.cudnn as cudnn
from dataset.deploy import DeployDataset
from network.textnet import TextNet
from cfglib.config import config as cfg
from cfglib.option import BaseOptions
from util.augmentation import BaseTransform
from util.visualize import visualize_gt
from util.misc import to_device, mkdirs, rescale_result
from pathlib import Path
import random
import timm
from torchvision import transforms
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel

# Removed duplicate imports and redundant lines


img_list = glob('/kaggle/input/totaltext-preprocessing/data/*.jpg')

input_dir = '/kaggle/input/totaltext-preprocessing/data'
extra_cfg = {
    'net': 'resnet18',
    'scale': 4,
    'exp_name': 'Totaltext',
    'checkepoch': 570,
    'test_size': [640, 960],
    'gpu': '1',
    'dis_threshold': 0.35,
    'cls_threshold': 0.9,
    'viz': True,
    'img_root': input_dir,
    'resume': None,
    'num_workers': 24,
    'cuda': False,
    'mgpu': False,
    'save_dir': './model/',
    'vis_dir': './vis/',
    'log_dir': './logs/',
    'loss': 'CrossEntropyLoss',
    'pretrain': False,
    'verbose': True,
    'max_epoch': 250,
    'lr': 1e-3,
    'lr_adjust': 'fix',
    'stepvalues': [],
    'weight_decay': 0.0,
    'gamma': 0.1,
    'momentum': 0.9,
    'batch_size': 6,
    'optim': 'Adam',
    'save_freq': 5,
    'display_freq': 10,
    'viz_freq': 50,
    'log_freq': 10000,
    'val_freq': 1000,
    'load_memory': False,
    'rescale': 255.0,
    'input_size': 640,
    'start_epoch': 0
}

def update_config(cfg, extra_cfg):
    for k, v in extra_cfg.items():
        cfg[k] = v
        # print(config.gpu)
        cfg.device = torch.device('cuda') if cfg.cuda else torch.device('cpu')

def osmkdir(input_pth):
    if os.path.isdir(input_pth):
        return input_pth
    
    input_dir = '/kaggle/working/input_dir'
    if os.path.exists(input_dir):
        try:
            shutil.rmtree(input_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            
    os.makedirs(input_dir)
    
    new_path = os.path.join(input_dir, Path(input_pth).name)
    shutil.copy2(input_pth, new_path)  # Use copy2 to preserve metadata
    
    # Return the path of the new folder
    return input_dir

def visualize_detection(image, output_dict, meta=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    cls_preds = F.interpolate(output_dict["fy_preds"], scale_factor=cfg.scale, mode='bilinear')
    cls_preds = cls_preds[0].data.cpu().numpy()

    py_preds = output_dict["py_preds"][1:]
    init_polys = output_dict["py_preds"][0]
    shows = []

    init_py = init_polys.data.cpu().numpy()
    path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                        meta['image_id'][0].split(".")[0] + "_init.png")

    im_show0 = image_show.copy()
    for i, bpts in enumerate(init_py.astype(np.int32)):
        cv2.drawContours(im_show0, [bpts.astype(np.int32)], -1, (255, 255, 0), 2)
        for j, pp in enumerate(bpts):
            if j == 0:
                cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (255, 0, 255), -1)
            elif j == 1:
                cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (0, 0, 255), -1)

    cv2.imwrite(path, im_show0)

    for idx, py in enumerate(py_preds):
        im_show = im_show0.copy()
        contours = py.data.cpu().numpy()
        cv2.drawContours(im_show, contours.astype(np.int32), -1, (0, 0, 255), 2)
        for ppts in contours:
            for j, pp in enumerate(ppts):
                if j == 0:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (0, 255, 255), -1)
                else:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (0, 255, 0), -1)
        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                             meta['image_id'][0].split(".")[0] + "_{}iter.png".format(idx))
        cv2.imwrite(path, im_show)
        shows.append(im_show)

    show_img = np.concatenate(shows, axis=1)
    show_boundary = cv2.resize(show_img, (320 * len(py_preds), 320))

    cls_pred = cav.heatmap(np.array(cls_preds[0] * 255, dtype=np.uint8))
    dis_pred = cav.heatmap(np.array(cls_preds[1] * 255, dtype=np.uint8))

#     heat_map = np.concatenate([cls_pred*255, dis_pred*255], axis=1)
    heat_map = cls_pred*255
    heat_map = cv2.resize(heat_map, (320, 320))

    return show_boundary, heat_map

def heatmap_to_binary(heatmap, threshold=127):
    heatmap_binary = np.where(heatmap[:,:, 0] >= threshold, 255, 0)
    
    return heatmap_binary

def inference_textBPN(model, test_loader):

    total_time = 0.
    art_results = {}
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        idx = 0  # test mode can only run with batch_size == 1
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
#         print(meta['image_id'], (H, W))

        input_dict['img'] = to_device(image)
        # get detection result
        start = time.time()
        output_dict = model(input_dict)
#         print(output_dict["py_preds"])
        torch.cuda.synchronize()
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0
        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
        
        gt_contour = []
        label_tag = meta['label_tag'][idx].int().cpu().numpy()
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())

        gt_vis = visualize_gt(img_show, gt_contour, label_tag)
        show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)
#             file_path = os.path.join(cfg.vis_dir, meta['image_id'][idx].split(".")[0]+"_heat_map.json")
        np.set_printoptions(threshold=np.inf)
#             print(heat_map[:,:, 0])
        heatmap_binary = heatmap_to_binary(heat_map)
#         plt.imshow(heatmap_binary)
#         plt.show()
        show_map = np.concatenate([heat_map, gt_vis], axis=1)
        show_map = cv2.resize(show_map, (320 * 3, 320))
        im_vis = np.concatenate([show_map, show_boundary], axis=0)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)
#         print('heatmap shape before resize', heatmap_binary.shape)
        
        heatmap_binary = cv2.resize(heatmap_binary.astype(float), (img_show.shape[1], img_show.shape[0]))
        image_text = np.zeros_like(img_show)
    
        for i in range(3):
            image_text[:, :, i] = np.where(heatmap_binary < 127, img_show[:, :, i], 0)
            
        result = {
            'heatmap': heatmap_binary,
            'contours': contours,
            'image_region': image_text
        }
#         print(result.keys())
        art_results[meta['image_id'][0]] = result
        
    return art_results

def focus_text(input_pth,model):
#     if not os.path.isdir(input_pth):
        
    extra_cfg['img_root'] = osmkdir(input_pth)
    update_config(cfg, extra_cfg)
    testset = DeployDataset(
        image_root=cfg.img_root,
        transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    )

    if cfg.cuda:
        cudnn.benchmark = True

    # Data
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    with torch.no_grad():
        res = inference_textBPN(model, test_loader)
        return  Image.fromarray(res[os.path.basename(input_pth)]['image_region'])

update_config(cfg, extra_cfg)
    
ocr_model = TextNet(is_training=False, backbone=cfg.net)
ocr_model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                          'TextBPN_{}_{}.pth'.format(ocr_model.backbone_name, cfg.checkepoch))

ocr_model.load_model(ocr_model_path)
ocr_model = ocr_model.to(cfg.device)  # copy to cuda
ocr_model.eval()   

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization for 3 channels 
])
model = timm.create_model('convnext_base.fb_in1k', pretrained=True)
num_classes = 3 # Make sure this is the correct number of classes in your model
in_features = model.get_classifier().in_features
model.fc = nn.Linear(in_features, num_classes)

# Define the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the checkpoint (your saved model)
checkpoint_path = "/kaggle/input/efficientnet/convnext.pth"  # Path to your checkpoint file
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load the model state_dict
model.load_state_dict(checkpoint)
model.reset_classifier(0)

font_model = model
font_model.to(device)

# Step 1: Load and Modify the Model
classes = ['horizontal','vertical','circular','curvy']
shape_model = timm.create_model('davit_small.msft_in1k', pretrained=False)
num_classes = len(classes)  # Set this to the number of classes in your dataset
in_features = shape_model.get_classifier().in_features
shape_model.fc = nn.Linear(in_features, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('/kaggle/input/davit-shape/model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the state dictionary into the model
shape_model.load_state_dict(checkpoint)
shape_model.to(device)
shape_model.eval()

def get_shape_embedding(image):
    '''
    Return shape embedding
    '''
    input_image = transform(image).unsqueeze(0)
    input_tensor = input_image.to(device)
    with torch.no_grad():
        embedding = shape_model(input_tensor).cpu()
        return embedding
    

# feature_extractor = FeatureExtractor(font_model)
def get_font_embedding(image):
    '''
    Return font embedding
    '''
    input_image = transform(image).unsqueeze(0)
    input_tensor = input_image.to(device)
    with torch.no_grad():
        embedding = font_model(input_tensor).cpu()
        return embedding


def get_color_embedding(input_image, n_clusters= 2):
    '''
    Return color embedding
    '''
    image = np.array(input_image.convert('RGB'))
    image_test = np.where(image > 5, image, 0)
    pixels = image_test.reshape(-1, 3)
    pixels = [x for x in pixels if sum(x) > 10]
    if len(pixels) == 0:
        feature = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        return feature
    kmeans = KMeans(n_init= 'auto', n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    dist = Counter(kmeans.labels_)
    proportion = np.array(list(dist.values()))
    proportion = proportion / sum(proportion)
    color_features = colors / 255.0
    proportion_rescaled = proportion * (1.0 / np.max(proportion))  # Normalize to [0, 1]
    proportion_features = np.repeat(proportion_rescaled, 3)  # Repeat proportion for each RGB component
    embedding = np.concatenate([color_features.flatten(), proportion_features.flatten()])
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalize to have equal scale
    
    return embedding

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

style_index= faiss.read_index('/kaggle/input/style-faiss-index/style_index.index')
with open('/kaggle/input/style-faiss-index/mapping.json', 'r') as f:
    mapping = json.load(f)
    
mapping = dict(mapping)
mapping = {int(key): value for key, value in mapping.items()}

num_embeddings = style_index.ntotal

# Retrieve all embeddings one by one
style_embeddings = [style_index.reconstruct(i) for i in range(num_embeddings)]

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
    dis_list = [distance(text_query.upper(), text.upper())/len(text_query) for text in text_list]
    
    # Convert to a numpy array for efficient sorting
    dis_array = np.array(dis_list)
    
    # Get sorted indices
    sorted_indices = np.argsort(dis_array)
    
    # Create sorted lists
    sorted_distances = dis_array[sorted_indices]
    sorted_text = [text_list[id] for id in sorted_indices]
    # print(sorted_text[:10])
    temp = [mapping[id] for id in sorted_indices]
   
    return  sorted_distances,sorted_indices

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

    plt.imshow(image)
    plt.show()
    
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
        
    print('font embedding:', font.shape)
    print('shape embedding:', shape.shape)
    print('color embedding:', color)

    if 'font' in feature_list:
        concat = np.concatenate([font*3, color, shape],axis = 1)
    else:
        concat = np.concatenate([font, color, shape],axis = 1)

    style_embed = []
    query_embed = np.array(concat)
    print(draf_embedding.shape, query_embed.shape)
    for embed in np.array(style_embeddings):
        new_embed = embed[query_embed[0] != None]
        style_embed.append(new_embed)
    style_embed = np.array(style_embed).astype('float32')
    print(style_embed.shape)

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
    return results
 

