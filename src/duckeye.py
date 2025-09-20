import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path
import os
from src.config import MODELS_PATH, SEGMENTATION_MODEL, BM_REGRESSOR_MODEL, PREPROCESSED_PATH, PLOT_PATH
import src.preprocessing.preprocessing as prep

class DuckEYE():

    def __init__(self, device:str):

        ### general ###
        self.device = device
        # self.input_dir = input_dir

        ### initialize models ###
        self.segmentation_model = torch.load(MODELS_PATH / SEGMENTATION_MODEL, weights_only=False).to(self.device)
        self.segmentation_model.eval()
        self.bm_regressor_model = torch.load(MODELS_PATH / BM_REGRESSOR_MODEL, weights_only=False).to(self.device)
        self.bm_regressor_model.eval()

        ### transformations ###
        self.segmentation_transforms = T.Compose([
            T.ToTensor()
        ])
        self.bm_regression_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ### density computation parameters ###
        self.lower_bound = np.array([30,40,40])
        self.upper_bound = np.array([90,255,255])

        print('initialization successfull')

    def compute_density(self, img):

        """
        run density calculation (not ML !)
        """
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, self.lower_bound, self.upper_bound)
        overlay = np.zeros_like(np.array(img))
        overlay[mask == 0] = [255, 0, 0] # red color for masked areas

        density = np.sum(mask==255) / mask.size
        return (density, overlay)

    def segment_density(self, image:Image.Image):

        """
        run density segmentation
        """
        image_tensor = self.segmentation_transforms(image).unsqueeze(0).to(self.device)
        output = self.segmentation_model(image_tensor)['out']
        mask = torch.argmax(output,dim=1).squeeze().cpu().numpy()
        overlay = np.zeros_like(np.array(image))
        overlay[mask > 0] = [255, 0, 0] # red for masked area
        density = np.sum(mask == 0) / mask.size

        return (density, overlay)

    def predict_biomass_density(self, image:Image.Image) -> float:
        
        """
        run biomass prediction
        """
        
        image_tensor = self.bm_regression_transforms(image).unsqueeze(0).to(self.device)
        output = self.bm_regressor_model(image_tensor).detach().cpu().numpy().ravel()[0]
        return output
    
    def observe_single(self, filename:str):

        """
        Process a single image and return results dict (for web JSON/response).
        """

        filename_original = filename
        filename_processed = filename.split('.')[0] + '_resized.jpg'

        full_path = PREPROCESSED_PATH / filename_processed

        img = cv2.imread(str(full_path))
        image_rgb = Image.open(full_path).convert('RGB')
        
        # Compute densities and prediction
        calculated_density, overlay_c = self.compute_density(img)
        segmented_density, overlay_s = self.segment_density(image_rgb)
        predicted_biomass_density = self.predict_biomass_density(image_rgb)
        
        # Generate plot (save to temp path)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted biomass: {predicted_biomass_density:.3f} g/m²")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(overlay_c, alpha=0.4)
        plt.title(f"Calculated Density: {calculated_density:.4f}")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image_rgb)
        plt.imshow(overlay_s, alpha=0.4)
        plt.title(f"Segmented Density: {segmented_density:.4f}")
        plt.axis('off')
        
        plot_path = PLOT_PATH / f"plot_{filename_original}"  # Save in input_dir or a temp dir
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return {
            'filename_original': filename_original,
            'filename_processed': filename_processed,
            'calculated_density': calculated_density,
            'segmented_density': segmented_density,
            'predicted_biomass': predicted_biomass_density,
            'plot_path': str(plot_path)  # Relative path for web serving
        }






        
