import timm
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import os

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
device = torch.device('cpu')
# map_location = torch.device('cpu') 


# Load the checkpoint (your saved model)
checkpoint_path = "./checkpoint/convnext.pth"  # Path to your checkpoint file
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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('./checkpoint/davit_shape', map_location= device)
# Load the state dictionary into the model
shape_model.load_state_dict(checkpoint)
shape_model.to(device)
shape_model.eval()

