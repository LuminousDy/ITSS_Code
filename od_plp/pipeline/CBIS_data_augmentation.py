import os
import cv2
import numpy as np
import random
import glob
from pathlib import Path
import shutil
from tqdm import tqdm
import albumentations as A

# Define path
processed_dataset_path = os.path.join('dataset', 'processed dataset', 'CBIS-DDSM')

def load_image_and_label(image_path, label_path):
    """
    Load image and corresponding label file
    """
    # Read image
    image = cv2.imread(image_path)
    
    # Read labels
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    return image, labels

def save_image_and_label(image, labels, image_path, label_path):
    """
    Save augmented image and labels
    """
    # Save image
    cv2.imwrite(image_path, image)
    
    # Save labels
    with open(label_path, 'w') as f:
        f.writelines(labels)

def apply_preprocessing(image):
    """
    Apply preprocessing
    1. Auto-orientation (EXIF orientation stripping) - handled by OpenCV read
    2. Resize to 416x416 (stretch)
    3. Auto-contrast via adaptive equalization
    """
    # Resize to 416x416
    image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale (if color image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for auto-contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # If original image was color, convert enhanced grayscale back to color format
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced

def random_crop(image, labels, max_crop_percent=0.2):
    """
    Randomly crop image 0-20%
    """
    height, width = image.shape[:2]
    
    # Determine crop percentage
    crop_percent = random.uniform(0, max_crop_percent)
    
    # Calculate crop pixels
    crop_pixels_h = int(height * crop_percent)
    crop_pixels_w = int(width * crop_percent)
    
    # Determine crop area
    crop_top = random.randint(0, crop_pixels_h)
    crop_bottom = random.randint(0, crop_pixels_h)
    crop_left = random.randint(0, crop_pixels_w)
    crop_right = random.randint(0, crop_pixels_w)
    
    # Calculate new boundaries
    new_top = crop_top
    new_bottom = height - crop_bottom
    new_left = crop_left
    new_right = width - crop_right
    
    # Ensure some image remains
    new_top = min(new_top, height - 10)
    new_left = min(new_left, width - 10)
    new_bottom = max(new_bottom, new_top + 10)
    new_right = max(new_right, new_left + 10)
    
    # Crop image
    cropped_image = image[new_top:new_bottom, new_left:new_right]
    
    # Adjust labels
    updated_labels = []
    for label in labels:
        parts = label.strip().split(' ')
        if len(parts) == 5:
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Adjust bounding box coordinates
            # Calculate new bounding box coordinates (relative to cropped image)
            new_width = new_right - new_left
            new_height = new_bottom - new_top
            
            # Original coordinates (pixels)
            orig_x_center = x_center * width
            orig_y_center = y_center * height
            orig_w = w * width
            orig_h = h * height
            
            # New coordinates (pixels)
            new_x_center = orig_x_center - new_left
            new_y_center = orig_y_center - new_top
            
            # Check if bounding box is still within image
            if (new_x_center + orig_w/2 > 0 and 
                new_x_center - orig_w/2 < new_width and 
                new_y_center + orig_h/2 > 0 and 
                new_y_center - orig_h/2 < new_height):
                
                # Crop bounding box (if it exceeds new image boundaries)
                x_min = max(0, new_x_center - orig_w/2)
                y_min = max(0, new_y_center - orig_h/2)
                x_max = min(new_width, new_x_center + orig_w/2)
                y_max = min(new_height, new_y_center + orig_h/2)
                
                # Calculate new center and width/height
                new_x_center = (x_min + x_max) / 2
                new_y_center = (y_min + y_max) / 2
                new_w = x_max - x_min
                new_h = y_max - y_min
                
                # Normalize
                new_x_center /= new_width
                new_y_center /= new_height
                new_w /= new_width
                new_h /= new_height
                
                # Keep if bounding box has minimum size
                if new_w > 0.01 and new_h > 0.01:
                    updated_label = f"{class_id} {new_x_center} {new_y_center} {new_w} {new_h}\n"
                    updated_labels.append(updated_label)
    
    # Resize image back to original size
    cropped_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_AREA)
    
    return cropped_image, updated_labels

def salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.05):
    """
    Apply salt and pepper noise to 5% of image pixels
    """
    # Create a copy of the image
    noisy = np.copy(image)
    
    # Add salt noise (white pixels)
    salt = np.ceil(amount * image.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    
    # Add pepper noise (black pixels)
    pepper = np.ceil(amount * image.size * (1. - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    
    return noisy

def additional_augmentations(image, labels):
    """
    Apply additional augmentations using Albumentations library
    """
    height, width = image.shape[:2]
    
    # Parse YOLO format labels to Albumentations bounding box format
    bboxes = []
    category_ids = []
    
    for label in labels:
        parts = label.strip().split(' ')
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            # Convert to Albumentations required [x_min, y_min, x_max, y_max] format
            x_min = (x_center - w/2)
            y_min = (y_center - h/2)
            x_max = (x_center + w/2)
            y_max = (y_center + h/2)
            
            bboxes.append([x_min, y_min, x_max, y_max])
            category_ids.append(class_id)
    
    # Define augmentation sequence - keeping only the most relevant for mammography
    transform = A.Compose([
        # Random rotation (max 15 degrees)
        A.Rotate(limit=15, p=0.5),
        
        # Random horizontal flip
        A.HorizontalFlip(p=0.5),
        
        # Random brightness and contrast adjustment
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Gaussian blur - useful for mammography to simulate image quality variations
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))
    
    # Apply augmentations
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    
    # Convert back to original format
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_category_ids = transformed['category_ids']
    
    # Convert augmented bounding boxes back to YOLO format
    transformed_labels = []
    for i, bbox in enumerate(transformed_bboxes):
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate center and width/height
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        
        class_id = transformed_category_ids[i]
        transformed_label = f"{class_id} {x_center} {y_center} {w} {h}\n"
        transformed_labels.append(transformed_label)
    
    return transformed_image, transformed_labels

def augment_dataset():
    """
    Augment the CBIS-DDSM dataset
    """
    # Only augment the training set
    train_images_dir = os.path.join(processed_dataset_path, 'train', 'images')
    train_labels_dir = os.path.join(processed_dataset_path, 'train', 'labels')
    
    # Ensure directories exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # Get all images
    image_paths = glob.glob(os.path.join(train_images_dir, '*.jpg'))
    
    print(f"Found {len(image_paths)} images in training set. Starting augmentation...")
    
    # Augment each image
    for image_path in tqdm(image_paths, desc="Augmenting images"):
        # Get corresponding label path
        base_name = os.path.basename(image_path).split('.')[0]
        label_path = os.path.join(train_labels_dir, f"{base_name}.txt")
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_path}")
            continue
        
        # Load image and labels
        image, labels = load_image_and_label(image_path, label_path)
        
        # Apply preprocessing
        processed_image = apply_preprocessing(image)
        
        # Create three versions of augmentation (consistent with CBIS-DDSM)
        for i in range(3):
            # Create new filename for augmented version
            aug_base_name = f"{base_name}_aug{i+1}"
            aug_image_path = os.path.join(train_images_dir, f"{aug_base_name}.jpg")
            aug_label_path = os.path.join(train_labels_dir, f"{aug_base_name}.txt")
            
            # Copy original image and labels as starting point
            aug_image = processed_image.copy()
            aug_labels = labels.copy()
            
            # Apply random crop (first augmentation method of CBIS-DDSM)
            aug_image, aug_labels = random_crop(aug_image, aug_labels)
            
            # Apply salt and pepper noise (second augmentation method of CBIS-DDSM)
            aug_image = salt_and_pepper_noise(aug_image)
            
            # Apply additional augmentations
            aug_image, aug_labels = additional_augmentations(aug_image, aug_labels)
            
            # Save augmented image and labels
            save_image_and_label(aug_image, aug_labels, aug_image_path, aug_label_path)
    
    print("Data augmentation completed!")

if __name__ == "__main__":
    print("Starting CBIS-DDSM dataset augmentation...")
    augment_dataset()
    print("CBIS-DDSM dataset augmentation completed!")
