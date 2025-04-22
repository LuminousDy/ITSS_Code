import os
import pandas as pd
import shutil
import numpy as np
from pathlib import Path
import random

# Define paths
original_dataset_path = os.path.join('dataset', 'original dataset', 'VinDr-Mammo')
processed_dataset_path = os.path.join('dataset', 'processed dataset', 'VinDr-Mammo')
images_source_path = os.path.join(original_dataset_path, 'Processed_Images')
metadata_file = os.path.join(original_dataset_path, 'updated_metadata.csv')

# Create necessary folder structure
def create_folders():
    # Create main folder
    os.makedirs(processed_dataset_path, exist_ok=True)
    
    # Create folders for train, validation, and test sets
    for split in ['train', 'valid', 'test']:
        for subfolder in ['images', 'labels']:
            os.makedirs(os.path.join(processed_dataset_path, split, subfolder), exist_ok=True)
    
    print("Folder structure created successfully.")

# Read and process metadata
def process_metadata():
    # Read metadata file
    df = pd.read_csv(metadata_file)
    
    # Filter images containing Mass or Suspicious Calcification
    filtered_df = df[df['finding_categories'].apply(lambda x: 'Mass' in x or 'Suspicious Calcification' in x)]
    
    # Print statistics
    print(f"Total images in metadata: {len(df)}")
    print(f"Filtered images (with Mass or Suspicious Calcification): {len(filtered_df)}")
    
    return filtered_df

# Convert coordinates to YOLO format
def convert_to_yolo_format(row):
    # Get image width and height
    img_width = row['width']
    img_height = row['height']
    
    # Get bounding box coordinates
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    
    # Calculate center coordinates and width/height
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Determine class ID: 0 for Calcification, 1 for Mass
    class_id = 1 if 'Mass' in row['finding_categories'] else 0
    
    # Return YOLO format annotation
    return f"{class_id} {x_center} {y_center} {width} {height}"

# Process dataset and split into train, validation, and test sets
def process_dataset(filtered_df):
    # Group data by study_id to ensure images of the same patient are in the same split
    study_ids = filtered_df['study_id'].unique()
    
    # Shuffle study_ids randomly
    random.seed(42)  # Set random seed for reproducibility
    random.shuffle(study_ids)
    
    # Allocate study_ids to train, validation, and test sets (70%, 15%, 15%)
    train_size = int(len(study_ids) * 0.7)
    valid_size = int(len(study_ids) * 0.15)
    
    train_ids = study_ids[:train_size]
    valid_ids = study_ids[train_size:train_size+valid_size]
    test_ids = study_ids[train_size+valid_size:]
    
    print(f"Train set: {len(train_ids)} studies")
    print(f"Validation set: {len(valid_ids)} studies")
    print(f"Test set: {len(test_ids)} studies")
    
    # Process and save images and annotations
    processed_count = 0
    for _, row in filtered_df.iterrows():
        study_id = row['study_id']
        image_id = row['image_id']
        
        # Determine split
        if study_id in train_ids:
            split = 'train'
        elif study_id in valid_ids:
            split = 'valid'
        else:
            split = 'test'
        
        # Create source image and destination paths
        src_img_path = os.path.join(images_source_path, study_id, f"{image_id}.png")
        dst_img_path = os.path.join(processed_dataset_path, split, 'images', f"{image_id}.png")
        
        # Create annotation file path
        label_path = os.path.join(processed_dataset_path, split, 'labels', f"{image_id}.txt")
        
        # Check if source image exists
        if os.path.exists(src_img_path):
            # Copy image
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create annotation file
            yolo_annotation = convert_to_yolo_format(row)
            with open(label_path, 'w') as f:
                f.write(yolo_annotation)
            
            processed_count += 1
            
            # Print progress
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images")
        else:
            print(f"Warning: Image not found - {src_img_path}")
    
    print(f"Successfully processed {processed_count} images and annotations.")

# Create dataset YAML file
def create_yaml_file():
    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['Calc', 'Mass']

"""
    
    with open(os.path.join(processed_dataset_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    print("Created data.yaml file.")

def main():
    print("Starting VinDr-Mammo dataset preprocessing...")
    
    # Create folder structure
    create_folders()
    
    # Process metadata
    filtered_df = process_metadata()
    
    # Process dataset
    process_dataset(filtered_df)
    
    # Create YAML file
    create_yaml_file()
    
    print("VinDr-Mammo dataset preprocessing completed!")

if __name__ == "__main__":
    main()
