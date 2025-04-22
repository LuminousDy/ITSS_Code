import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import time


def parse_args():
    """
    Parse command line arguments for mammography object detection
    """
    parser = argparse.ArgumentParser(description='Mammography Object Detection with YOLOv8')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='runs/detect/YOLOv8_20250422_122835/weights/best.pt', 
                        help='Path to model weights (default: runs/detect/YOLOv8_20250422_122835/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--device', type=str, default='', help='Device to run on (empty for auto, cpu, 0, 1, etc.)')
    parser.add_argument('--save', action='store_true', help='Save results to sample/results/')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections')
    
    return parser.parse_args()


def load_model(model_path, device=''):
    """
    Load YOLOv8 model from path
    """
    try:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def detect_objects(model, image_path, conf=0.25, iou=0.7, imgsz=640, device='', max_det=300):
    """
    Detect objects in the image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"Running detection on {image_path}...")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf,         # Confidence threshold
        iou=iou,           # IoU threshold for NMS
        imgsz=imgsz,       # Image size
        device=device,     # Device (empty string means auto)
        max_det=max_det,   # Maximum detections
        verbose=False
    )[0]  # Get the first result (only one image)
    
    return results


def process_results(results, image_path):
    """
    Process and visualize detection results
    """
    # Load original image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    
    # Create a copy for drawing
    image_with_boxes = image.copy()
    
    # Extract detection information
    boxes = results.boxes.xyxy.cpu().numpy()  # Get bounding boxes
    confidences = results.boxes.conf.cpu().numpy()  # Get confidence scores
    class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Get class IDs
    
    # Get class names
    class_names = results.names
    
    # Print detection information
    print(f"\nDetection Results for {os.path.basename(image_path)}:")
    print(f"Found {len(boxes)} objects")
    
    detections = []
    
    # Draw bounding boxes and labels on the image
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[cls_id]} {conf:.2f}"
        
        # Choose a color based on class
        color = (255, 0, 0) if cls_id == 0 else (0, 255, 0)  # Red for class 0, Green for class 1
        
        # Draw box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Add label with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image_with_boxes, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Store detection information
        detections.append({
            'class': class_names[cls_id],
            'confidence': float(conf),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
        
        print(f"  {i+1}. {class_names[cls_id]}: {conf:.4f}, Box: [{x1}, {y1}, {x2}, {y2}]")
    
    return image_with_boxes, detections


def save_results(image_with_boxes, detections, image_path, output_dir='sample/results'):
    """
    Save results to disk
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file names
    image_name = Path(image_path).stem
    output_image_path = os.path.join(output_dir, f"{image_name}_detection.jpg")
    output_text_path = os.path.join(output_dir, f"{image_name}_detection.txt")
    
    # Save image with detections
    plt.figure(figsize=(12, 12))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save detection information to text file
    with open(output_text_path, 'w') as f:
        f.write(f"Detection results for {os.path.basename(image_path)}:\n")
        f.write("-" * 50 + "\n")
        for i, det in enumerate(detections):
            f.write(f"{i+1}. Class: {det['class']}, Confidence: {det['confidence']:.4f}\n")
            f.write(f"   Bounding Box: [x1={det['bbox'][0]}, y1={det['bbox'][1]}, x2={det['bbox'][2]}, y2={det['bbox'][3]}]\n")
            f.write("\n")
    
    print(f"Results saved to {output_dir}")
    print(f"  - Detection image: {output_image_path}")
    print(f"  - Detection data: {output_text_path}")


def display_results(image_with_boxes):
    """
    Display the image with detections
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to process a mammography image and detect abnormalities
    """
    # Parse arguments
    args = parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Start timing
    start_time = time.time()
    
    # Detect objects
    results = detect_objects(
        model=model,
        image_path=args.image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det
    )
    
    # Process results
    image_with_boxes, detections = process_results(results, args.image)
    
    # End timing
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Save results if requested
    if args.save:
        save_results(image_with_boxes, detections, args.image)
    
    # Display results if requested
    if args.show:
        display_results(image_with_boxes)
    
    return detections


def infer_single_image(image_path, model_path=None, conf=0.25, show=False, save=False):
    """
    Function for programmatic inference on a single image
    
    Args:
        image_path (str): Path to the image
        model_path (str, optional): Path to model weights
        conf (float, optional): Confidence threshold
        show (bool, optional): Whether to display results
        save (bool, optional): Whether to save results
        
    Returns:
        list: List of detections with class, confidence and bounding box
    """
    # Use default model path if not specified
    if model_path is None:
        model_path = 'runs/detect/YOLOv8_20250422_122835/weights/best.pt'
    
    # Load model
    model = load_model(model_path)
    
    # Detect objects
    results = detect_objects(
        model=model,
        image_path=image_path,
        conf=conf
    )
    
    # Process results
    image_with_boxes, detections = process_results(results, image_path)
    
    # Save results if requested
    if save:
        save_results(image_with_boxes, detections, image_path)
    
    # Display results if requested
    if show:
        display_results(image_with_boxes)
    
    return detections


if __name__ == "__main__":
    detections = main()
