import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# Define a custom Dataset class
class FontDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.images_path = []
        self.labels = []

        for _, row in self.data.iterrows():
            img_path = row["image_path"]
            font = row["font"]

            self.images_path.append(img_path)
            self.labels.append(font)
        
        self.font_mapping = pd.Categorical(self.labels)
        self.labels = self.font_mapping.codes

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        return self.images_path[idx], self.labels[idx]
    
    def get_font_name(self, label):
        """
        Convert a label (integer) back to the font name.
        """ 
        return self.font_mapping.categories[label]
    
# Define model
class FontClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FontClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjusted dimensions for 56x56 feature map
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x