import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_args():
    """
    Parse command line arguments for mammography image classification
    """
    parser = argparse.ArgumentParser(description='Mammography Image Classification with EfficientNetV2B0')
    parser.add_argument('--image', type=str, default='sample/sample_data/sample_image.jpg', help='Path to input image')
    parser.add_argument('--model', type=str, default='runs/classification/IC_20250427_170000/weights/best.keras', 
                        help='Path to model weights (default: runs/classification/IC_20250427_170000/weights/best.keras)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for classification')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size for inference (224 for EfficientNetV2B0)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (empty for auto, cpu, 0, 1, etc.)')
    parser.add_argument('--save', action='store_true', help='Save results to sample/results/')
    parser.add_argument('--show', action='store_true', help='Display results')
    
    return parser.parse_args()


def load_model(model_path, device=''):
    """
    Load the classification model from path
    """
    try:
        print(f"Loading model from {model_path}...")
        
        # Verify file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Print file size for debugging
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"Model file size: {file_size:.2f} MB")
        
        # Set device if specified
        if device:
            if device.lower() == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = device
        
        # Check file extension
        file_extension = os.path.splitext(model_path)[1].lower()
        print(f"Detected file extension: {file_extension}")
        
        # Try multiple loading approaches
        if file_extension == '.pt' or file_extension == '.pth':
            # For PyTorch models
            print("Detected PyTorch model format (.pt/.pth)")
            try:
                import torch
                model = torch.load(model_path)
                return model
            except Exception as e:
                print(f"Failed to load as PyTorch model: {e}")
                raise
                
        elif file_extension == '.h5':
            # For older Keras H5 format
            print("Detected Keras H5 format (.h5)")
            try:
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                print(f"Failed to load as H5 model: {e}")
                # Try alternative method
                try:
                    print("Attempting to load model with custom_objects...")
                    model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects={'KerasLayer': lambda x: x}
                    )
                    return model
                except Exception as e2:
                    print(f"Alternative H5 loading failed: {e2}")
                    raise e
        
        elif file_extension == '.keras':
            # For newer Keras format
            print("Detected newer Keras format (.keras)")
            try:
                # Check if the file might be corrupted
                if file_size < 0.01:  # Very small file
                    print("Warning: File is suspiciously small, might be corrupted")
                
                # Try with compile=False first
                print("Attempting to load with compile=False...")
                model = tf.keras.models.load_model(model_path, compile=False)
                return model
            except Exception as e:
                print(f"Failed to load .keras model: {e}")
                try:
                    # Try with tf.io.gfile.GFile for better error handling
                    print("Attempting to load with tf.io.gfile...")
                    with tf.io.gfile.GFile(model_path, 'rb') as f:
                        model_content = f.read()
                        if len(model_content) < 100:  # Very small file
                            print(f"Warning: File content is only {len(model_content)} bytes, likely corrupted")
                    
                    # Try the parent directory as SavedModel format
                    print("Attempting to load from parent directory as SavedModel...")
                    model_dir = os.path.dirname(model_path)
                    if os.path.exists(os.path.join(model_dir, 'saved_model.pb')):
                        model = tf.keras.models.load_model(model_dir)
                        return model
                    
                    # Try alternate path - look for .h5 file instead
                    alt_path = model_path.replace('.keras', '.h5')
                    if os.path.exists(alt_path):
                        print(f"Found alternative .h5 file, trying to load: {alt_path}")
                        model = tf.keras.models.load_model(alt_path)
                        return model
                    
                    # Look for any model files in the directory
                    model_dir = os.path.dirname(model_path)
                    print(f"Looking for alternate model files in: {model_dir}")
                    for filename in os.listdir(model_dir):
                        if filename.endswith(('.h5', '.keras', '.pb')):
                            full_path = os.path.join(model_dir, filename)
                            print(f"Found potential model file: {full_path}")
                            try:
                                model = tf.keras.models.load_model(full_path)
                                print(f"Successfully loaded alternative model: {full_path}")
                                return model
                            except Exception as e_alt:
                                print(f"Failed to load alternative model {full_path}: {e_alt}")
                    
                    raise e
                except Exception as e2:
                    print(f"All .keras loading alternatives failed: {e2}")
                    raise e
        
        elif file_extension == '' and os.path.isdir(model_path):
            # For SavedModel directory format
            print("Detected potential SavedModel directory format")
            if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                model = tf.keras.models.load_model(model_path)
                return model
            else:
                print(f"No saved_model.pb found in directory: {model_path}")
                # List directory contents for debugging
                print(f"Directory contents: {os.listdir(model_path)}")
                raise FileNotFoundError(f"No saved_model.pb found in: {model_path}")
            
        else:
            # Generic case - try standard loading
            print(f"Attempting to load model with auto-detection...")
            try:
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                print(f"Standard loading failed: {e}")
                # Try converting the model path
                if 'classification' not in model_path and 'detect' in model_path:
                    # Try alternate path convention
                    alt_path = model_path.replace('detect', 'classification')
                    print(f"Trying alternate path: {alt_path}")
                    if os.path.exists(alt_path):
                        model = tf.keras.models.load_model(alt_path)
                        return model
                raise
                
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a fallback model for debugging
        print("Creating a simple fallback model for debugging...")
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("WARNING: Using fallback model since the actual model could not be loaded!")
        return model


def classify_image(model, image_path, imgsz=224, conf=0.5):
    """
    Classify the image using the loaded model
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Running classification on {image_path}...")
    
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB (important for models trained on RGB images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to model's expected size
    image = cv2.resize(image, (imgsz, imgsz))
    
    # Normalize pixel values to [0, 1]
    # image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Run inference
    start_time = time.time()
    predictions = model.predict(image, verbose=0)
    inference_time = time.time() - start_time
    
    # Get probabilities
    probabilities = predictions[0] * 100
    
    # Get predicted class (0: Benign, 1: Malignant)
    predicted_class = np.argmax(probabilities)
    
    # Only consider as positive if above confidence threshold
    class_confidence = probabilities[predicted_class]
    if class_confidence < conf * 100:
        predicted_class = -1  # Not confident enough
    
    result = {
        'class_id': predicted_class,
        'class_name': 'Benign' if predicted_class == 0 else 'Malignant' if predicted_class == 1 else 'Unknown',
        'confidence': class_confidence / 100,  # Convert back to [0, 1]
        'probabilities': {
            'Benign': probabilities[0] / 100,
            'Malignant': probabilities[1] / 100
        },
        'inference_time': inference_time
    }
    
    return result


def process_results(result, image_path):
    """
    Process and visualize classification results
    """
    # Load original image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    
    # Create a copy for drawing
    image_with_class = image.copy()
    
    # Get the dimensions of the image
    h, w = image.shape[:2]
    
    # Define text parameters
    text = f"{result['class_name']}: {result['confidence']:.2%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Compute text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text at the top of the image
    text_x = 10
    text_y = text_size[1] + 20
    
    # Draw a colored background based on class
    color = (0, 200, 0) if result['class_name'] == 'Benign' else (200, 0, 0) if result['class_name'] == 'Malignant' else (100, 100, 100)
    
    # Draw background rectangle
    cv2.rectangle(image_with_class, (0, 0), (w, 40 + text_size[1]), color, -1)
    
    # Add text
    cv2.putText(image_with_class, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Add confidence values as a separate text below
    conf_text = f"Benign: {result['probabilities']['Benign']:.2%}, Malignant: {result['probabilities']['Malignant']:.2%}"
    conf_text_y = text_y + 30
    cv2.putText(image_with_class, conf_text, (text_x, conf_text_y), font, 0.6, (255, 255, 255), 1)
    
    # Print classification information
    print(f"\nClassification Results for {os.path.basename(image_path)}:")
    print(f"Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Benign Probability: {result['probabilities']['Benign']:.2%}")
    print(f"Malignant Probability: {result['probabilities']['Malignant']:.2%}")
    print(f"Inference time: {result['inference_time']:.2f} seconds")
    
    return image_with_class


def save_results(image_with_class, result, image_path, output_dir='sample/results'):
    """
    Save results to disk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file names
    image_name = Path(image_path).stem
    output_image_path = os.path.join(output_dir, f"{image_name}_classification.jpg")
    output_text_path = os.path.join(output_dir, f"{image_name}_classification.txt")
    
    # Save image with classification
    plt.figure(figsize=(12, 12))
    plt.imshow(image_with_class)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create classification-style format to match od_main.py
    classification_format = [{
        'class': result['class_name'],
        'confidence': result['confidence'],
    }]
    
    # Save classification information to text file in a format similar to od_main.py
    with open(output_text_path, 'w') as f:
        f.write(f"Classification results for {os.path.basename(image_path)}:\n")
        f.write("-" * 50 + "\n")
        f.write(f"1. Class: {result['class_name']}, Confidence: {result['confidence']:.4f}\n")
        f.write(f"   Benign Probability: {result['probabilities']['Benign']:.4f}, Malignant Probability: {result['probabilities']['Malignant']:.4f}\n")
        f.write(f"   Inference time: {result['inference_time']:.2f} seconds\n")
        f.write("\n")
    
    print(f"Results saved to {output_dir}")
    print(f"  - Classification image: {output_image_path}")
    print(f"  - Classification data: {output_text_path}")


def display_results(image_with_class):
    """
    Display the image with classification result
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_class)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to process a mammography image and classify it
    """
    # Parse arguments
    args = parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Start timing
    start_time = time.time()
    
    # Classify image
    result = classify_image(
        model=model,
        image_path=args.image,
        imgsz=args.imgsz,
        conf=args.conf
    )
    
    # Process results
    image_with_class = process_results(result, args.image)
    
    # End timing
    total_time = time.time() - start_time
    print(f"Classification completed in {total_time:.2f} seconds")
    
    # Always save results, regardless of --save parameter
    save_results(image_with_class, result, args.image)
    
    # Display results if requested
    if args.show:
        display_results(image_with_class)
    
    return result


def infer_single_image(image_path, model_path=None, conf=0.5, show=False):
    """
    Function for programmatic inference on a single image
    
    Args:
        image_path (str): Path to the image
        model_path (str, optional): Path to model weights
        conf (float, optional): Confidence threshold
        show (bool, optional): Whether to display results
        
    Returns:
        dict: Classification result with class, confidence and probabilities
    """
    # Use default model path if not specified
    if model_path is None:
        model_path = 'runs/classification/IC_20250427_170000/weights/best.keras'
    
    # Load model
    model = load_model(model_path)
    
    # Classify image
    result = classify_image(
        model=model,
        image_path=image_path,
        conf=conf
    )
    
    # Process results
    image_with_class = process_results(result, image_path)
    
    # Always save results
    save_results(image_with_class, result, image_path)
    
    # Display results if requested
    if show:
        display_results(image_with_class)
    
    return result


if __name__ == "__main__":
    result = main()
