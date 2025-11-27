import torch
import json
import cv2
import os
import numpy as np
from torch.utils.data import Dataset

class RunwayKeypointDataset(Dataset):
    def __init__(self, img_dir, json_path):
        self.img_dir = img_dir
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        self.filenames = list(self.labels.keys())
        
        self.line_order = ['LEDG', 'REDG', 'CTL']
        
        self.width = 640
        self.height = 360

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 1. Load Image
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Extract Coordinates (No Scaling Needed!)
        raw_points = []
        image_data = self.labels[fname]
        lines_dict = {item['label']: item['points'] for item in image_data}
        
        for label in self.line_order:
            if label in lines_dict:
                pts = lines_dict[label]
                raw_points.extend(pts[0]) # Start point [x, y]
                raw_points.extend(pts[1]) # End point [x, y]
            else:
                raw_points.extend([0, 0, 0, 0]) # Handle missing lines

        coords = np.array(raw_points, dtype=np.float32)

        coords[0::2] /= self.width  
        coords[1::2] /= self.height 
        
        # 4. To Tensor
        image = image.transpose(2, 0, 1) # HWC -> CHW
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        target = torch.tensor(coords, dtype=torch.float32)

        return image, target