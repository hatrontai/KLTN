from util.model import font_model, shape_model, transform
import torch
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import faiss
import json
from PIL import Image
import cv2


device = torch.device('cpu')

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


def get_dominant_colors(pil_img, palette_size=16, num_colors=10):
    # Resize image to speed up processing
    # img = Image.open(pil_img_path)
    img = pil_img.copy()
    # plt.imshow(img)
    # plt.show()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    # print(color_counts)
    dominant_colors = []
    for i in range(num_colors):
        if i >= len(color_counts):
            return [[0 for _ in range(3)] for _ in range(num_colors)]
        palette_index = color_counts[i][1]
        dominant_colors.append(palette[palette_index*3:palette_index*3+3])

    return dominant_colors
    
def cosine_similarity(vector_a, vector_b):
    vector_a = np.squeeze(vector_a)
    vector_b = np.squeeze(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_color_embedding(input_image, n_clusters= 2):
    '''
    Return color embedding
    '''
    image = np.array(input_image.convert('RGB'))
    colors = get_dominant_colors(input_image, palette_size=8, num_colors= 3)
    # print(colors)
    
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Kiểm tra xem ảnh có alpha channel không
    if image.shape[2] == 4:  # RGBA
        # Tạo mask từ alpha channel (nền đã được xóa)
        alpha_channel = image[:, :, 3]
        _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    else:
        # Chuyển ảnh RGB/Grayscale thành ảnh nhị phân
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Tìm contour
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vẽ contour lên ảnh mới
    contour_image = np.zeros_like(binary_mask)
    # cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)
    
    #######
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Vẽ vùng contour kín
    
    # Sử dụng erosion để thu nhỏ vùng về phía trong
    kernel = np.ones((8, 8), np.uint8)  # Kích thước kernel (điều chỉnh độ co)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Tìm contour mới sau khi erosion
    eroded_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vẽ contour mới lên ảnh
    # contour_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, eroded_contours, -1, (255), thickness=2) 
    ######
    
    # Lưu kết quả hoặc hiển thị
    # plt.imshow(image_rgb)
    # plt.show()
    
    # print(contour_image.shape)
    contour = image
    for i in range(3):
        contour[:,:,i] = np.where(contour_image > 0, image[:,:,i], 0)
    
    # plt.imshow(contour)
    # plt.show()
    non_black_pixels = contour.reshape(-1, 3)  # Chỉ lấy các pixel không phải màu đen
    non_black_pixels = [x for x in non_black_pixels if sum(x) > 1]
    if len(non_black_pixels) == 0:
        return np.array([0, 0, 0])
    # Bước 1: Chuyển list thành mảng NumPy
    pixel_array = np.array(non_black_pixels)
    
    # Bước 2: Áp dụng KMeans để phân cụm thành 2 cụm
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(pixel_array)
    
    # Bước 3: Lấy nhãn của từng pixel và đếm số lượng trong mỗi cụm
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Bước 4: Xác định cụm có số lượng pixel lớn nhất
    largest_cluster_label = unique_labels[np.argmax(counts)]
    
    # Bước 5: Lấy màu trung bình của cụm lớn nhất
    largest_cluster_pixels = pixel_array[labels == largest_cluster_label]
    color_br = [np.mean(largest_cluster_pixels, axis=0).astype(int)]

    colors = colors[1:]
    distances = []
    for i, color in enumerate(colors):
        distances.append(cosine_similarity(color, color_br))
    index_color_text = distances.index(min(distances))
    # print(distances, index_color_text)
    embedding = np.array(colors[index_color_text]) / 255.0

    return embedding

def load_style_embedding():
    style_index= faiss.read_index('./embedding/style_index.index')
    with open('./embedding/mapping.json', 'r') as f:
        mapping = json.load(f)
        
    mapping = dict(mapping)
    mapping = {int(key): value for key, value in mapping.items()}

    num_embeddings = style_index.ntotal

    # Retrieve all embeddings one by one
    style_embeddings = [style_index.reconstruct(i) for i in range(num_embeddings)]
    return style_embeddings, mapping