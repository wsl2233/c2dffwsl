import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class AdaptiveLightAwareMask:
    """
    Adaptive Light-Aware Mask (ALM) for multimodal remote sensing images
    """
    
    def __init__(self, patch_size=16, k=10, p=0.5):
        self.patch_size = patch_size
        self.k = k  # number of patches to select
        self.p = p  # probability for RGB masking in bright patches
        
    def calculate_brightness(self, image):
        """Calculate brightness of an image"""
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def get_patch_brightness(self, image):
        """Calculate brightness for each patch"""
        h, w = image.shape[:2]
        patch_h = h // self.patch_size
        patch_w = w // self.patch_size
        
        brightness_map = []
        
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                patch = image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                brightness = self.calculate_brightness(patch)
                brightness_map.append(brightness)
        
        return np.array(brightness_map)
    
    def apply_alm(self, rgb_image, ir_image):
        """Apply Adaptive Light-Aware Mask"""
        # Convert images to numpy arrays if they are tensors
        if torch.is_tensor(rgb_image):
            rgb_image = rgb_image.numpy().transpose(1, 2, 0)
        if torch.is_tensor(ir_image):
            ir_image = ir_image.numpy().squeeze()
        
        # Calculate brightness for RGB image
        brightness_map = self.get_patch_brightness(rgb_image)
        
        # Sort patches by brightness
        sorted_indices = np.argsort(brightness_map)
        
        # Select top-k dark patches and top-k bright patches
        dark_patches = sorted_indices[:self.k]
        bright_patches = sorted_indices[-self.k:]
        
        # Create augmented images
        augmented_rgb = rgb_image.copy()
        augmented_ir = ir_image.copy()
        
        h, w = rgb_image.shape[:2]
        patch_height = h // self.patch_size
        patch_width = w // self.patch_size
        
        # Apply masks to dark patches (always mask RGB)
        for idx in dark_patches:
            i, j = divmod(idx, self.patch_size)
            augmented_rgb[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width] = 0
        
        # Apply masks to bright patches
        # Randomly select p*k patches to mask RGB, remaining mask IR
        num_rgb_masks = int(self.p * self.k)
        random_indices = np.random.permutation(self.k)
        
        rgb_mask_indices = bright_patches[random_indices[:num_rgb_masks]]
        ir_mask_indices = bright_patches[random_indices[num_rgb_masks:]]
        
        for idx in rgb_mask_indices:
            i, j = divmod(idx, self.patch_size)
            augmented_rgb[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width] = 0
        
        for idx in ir_mask_indices:
            i, j = divmod(idx, self.patch_size)
            augmented_ir[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width] = 0
        
        return augmented_rgb, augmented_ir

def main():
    """Example usage"""
    alm = AdaptiveLightAwareMask()
    
    # Example: Load your RGB and IR images
    # rgb_image = cv2.imread('path/to/rgb.jpg')
    # ir_image = cv2.imread('path/to/ir.jpg')
    
    # Apply ALM
    # augmented_rgb, augmented_ir = alm.apply_alm(rgb_image, ir_image)
    
    print("ALM module loaded successfully!")

if __name__ == "__main__":
    main()