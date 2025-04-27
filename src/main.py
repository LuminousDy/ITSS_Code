import os
import argparse
import sys
from pathlib import Path

# Add the root directory to the Python path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the three modules
from od_plp.od_main import load_model as load_od_model
from od_plp.od_main import detect_objects, process_results as process_od_results
from od_plp.od_main import save_results as save_od_results, display_results as display_od_results

from ic_plp.ic_main import load_model as load_ic_model
from ic_plp.ic_main import classify_image, process_results as process_ic_results
from ic_plp.ic_main import save_results as save_ic_results, display_results as display_ic_results

from vlm_plp.vlm_main import load_model as setup_vlm_api
from vlm_plp.vlm_main import read_detection_results, read_classification_results
from vlm_plp.vlm_main import construct_medical_report_prompt, process_image as process_vlm_image
from vlm_plp.vlm_main import save_results as save_vlm_results, display_results as display_vlm_results


def parse_args():
    """
    Parse command line arguments for the mammography analysis pipeline
    """
    parser = argparse.ArgumentParser(
        description='Mammography Analysis Pipeline - Object Detection, Classification, and Medical Report Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments for all modes
    parser.add_argument('--image', type=str, default='sample/sample_data/sample_image.jpg', 
                        help='Path to input image')
    parser.add_argument('--save', action='store_true', help='Save results to sample/results/')
    parser.add_argument('--show', action='store_true', help='Display results')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Object Detection (OD) subparser
    od_parser = subparsers.add_parser('od', help='Object Detection mode')
    od_parser.add_argument('--model', type=str, default='runs/detect/YOLOv8_20250422_122835/weights/best.pt', 
                        help='Path to OD model weights')
    od_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    od_parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    od_parser.add_argument('--imgsz', type=int, default=640, help='Image size for detection inference')
    od_parser.add_argument('--device', type=str, default='', help='Device to run detection on')
    od_parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections')
    
    # Image Classification (IC) subparser
    ic_parser = subparsers.add_parser('ic', help='Image Classification mode')
    ic_parser.add_argument('--model', type=str, default='runs/classification/IC_20250427_170000/weights/best.keras', 
                        help='Path to IC model weights')
    ic_parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for classification')
    ic_parser.add_argument('--imgsz', type=int, default=224, help='Image size for classification inference')
    ic_parser.add_argument('--device', type=str, default='', help='Device to run classification on')
    
    # Vision Language Model (VLM) Medical Report subparser
    vlm_parser = subparsers.add_parser('vlm', help='VLM Medical Report Generation mode')
    vlm_parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B', help='VLM model name')
    vlm_parser.add_argument('--detection-file', type=str, default='', help='Path to detection results file (if not provided, will run OD first)')
    vlm_parser.add_argument('--classification-file', type=str, default='', help='Path to classification results file (if not provided, will run IC first)')
    vlm_parser.add_argument('--custom-prompt', type=str, default='', help='Custom prompt to use instead of the default medical report prompt')
    
    # Full pipeline mode
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline (OD, IC, and VLM)')
    pipeline_parser.add_argument('--od-model', type=str, default='runs/detect/YOLOv8_20250422_122835/weights/best.pt', 
                        help='Path to OD model weights')
    pipeline_parser.add_argument('--ic-model', type=str, default='runs/classification/IC_20250427_170000/weights/best.keras', 
                        help='Path to IC model weights')
    pipeline_parser.add_argument('--vlm-model', type=str, default='Qwen2.5-VL-7B', help='VLM model name')
    pipeline_parser.add_argument('--od-conf', type=float, default=0.25, help='Confidence threshold for detections')
    pipeline_parser.add_argument('--ic-conf', type=float, default=0.5, help='Confidence threshold for classification')
    pipeline_parser.add_argument('--device', type=str, default='', help='Device to run inference on')
    
    return parser.parse_args()


def run_object_detection(args):
    """
    Run the object detection pipeline
    """
    print("\n=== Running Object Detection ===")
    
    # Load the model
    model = load_od_model(args.model, args.device)
    
    # Perform detection
    results = detect_objects(
        model=model,
        image_path=args.image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det
    )
    
    # Process and visualize results
    image_with_boxes, detections = process_od_results(results, args.image)
    
    # Save results
    save_od_results(image_with_boxes, detections, args.image)
    
    # Display results if requested
    if args.show:
        display_od_results(image_with_boxes)
    
    # Generate file paths even though the save_od_results function doesn't return them
    output_dir = 'sample/results'
    image_name = Path(args.image).stem
    detection_image = os.path.join(output_dir, f"{image_name}_detection.jpg")
    detection_data = os.path.join(output_dir, f"{image_name}_detection.txt")
    
    return detections, [detection_image, detection_data]


def run_image_classification(args):
    """
    Run the image classification pipeline
    """
    print("\n=== Running Image Classification ===")
    
    # Load the model
    model = load_ic_model(args.model, args.device)
    
    # Perform classification
    result = classify_image(
        model=model,
        image_path=args.image,
        imgsz=args.imgsz,
        conf=args.conf
    )
    
    # Process and visualize results
    image_with_class = process_ic_results(result, args.image)
    
    # Save results
    save_ic_results(image_with_class, result, args.image)
    
    # Display results if requested
    if args.show:
        display_ic_results(image_with_class)
    
    # Generate file paths even though the save_ic_results function doesn't return them
    output_dir = 'sample/results'
    image_name = Path(args.image).stem
    classification_image = os.path.join(output_dir, f"{image_name}_classification.jpg")
    classification_data = os.path.join(output_dir, f"{image_name}_classification.txt")
    
    return result, [classification_image, classification_data]


def run_vlm_medical_report(args, detection_file=None, classification_file=None):
    """
    Run the VLM medical report generation pipeline
    """
    print("\n=== Running VLM Medical Report Generation ===")
    
    # Set up API connection
    if not setup_vlm_api():
        print("Failed to set up VLM API connection. Exiting.")
        return None, None
    
    # Determine detection and classification file paths
    det_file = args.detection_file if hasattr(args, 'detection_file') and args.detection_file else detection_file
    cls_file = args.classification_file if hasattr(args, 'classification_file') and args.classification_file else classification_file
    
    # Ensure the files exist
    if not det_file or not os.path.exists(det_file):
        print(f"Error: Detection file not found. Path: {det_file}")
        return None, None
    
    if not cls_file or not os.path.exists(cls_file):
        print(f"Error: Classification file not found. Path: {cls_file}")
        return None, None
    
    print(f"Using detection file: {det_file}")
    print(f"Using classification file: {cls_file}")
    
    # Read detection and classification results
    detection_results = read_detection_results(det_file)
    classification_results = read_classification_results(cls_file)
    
    # Construct the prompt
    if hasattr(args, 'custom_prompt') and args.custom_prompt:
        prompt = args.custom_prompt
        print("Using custom prompt")
    else:
        prompt = construct_medical_report_prompt(detection_results, classification_results)
        print("Using default medical report prompt")
    
    # Save the prompt for inspection
    output_dir = 'sample/results'
    os.makedirs(output_dir, exist_ok=True)
    prompt_file = os.path.join(output_dir, "medical_report_prompt.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Prompt saved to {prompt_file} for inspection")
    
    # Process image
    model_name = args.model if hasattr(args, 'model') and args.model else args.vlm_model if hasattr(args, 'vlm_model') else "Qwen2.5-VL-7B"
    print(f"Using VLM model: {model_name}")
    
    response, processing_time = process_vlm_image(
        image_path=args.image,
        model_name=model_name,
        prompt=prompt
    )
    
    if response:
        # Save results
        report_path = save_vlm_results(
            response=response,
            image_path=args.image,
            prompt=prompt,
            processing_time=processing_time
        )
        
        # Display results if requested
        if args.show:
            display_vlm_results(args.image, report_path)
        
        return response, report_path
    
    return None, None


def run_full_pipeline(args):
    """
    Run the full pipeline: object detection, image classification, and VLM medical report generation
    """
    print("\n=== Running Full Mammography Analysis Pipeline ===")
    
    # 1. Run Object Detection
    print("\nStep 1: Object Detection")
    od_args = argparse.Namespace(
        image=args.image,
        model=args.od_model,
        conf=args.od_conf,
        iou=0.7,
        imgsz=640,
        device=args.device,
        max_det=300,
        save=args.save,
        show=False  # Don't show intermediate results in pipeline mode
    )
    detections, od_files = run_object_detection(od_args)
    
    # Check if detection was successful
    if not detections or not od_files:
        print("Error: Object detection failed.")
        return
    
    # 2. Run Image Classification
    print("\nStep 2: Image Classification")
    ic_args = argparse.Namespace(
        image=args.image,
        model=args.ic_model,
        conf=args.ic_conf,
        imgsz=224,
        device=args.device,
        save=args.save,
        show=False  # Don't show intermediate results in pipeline mode
    )
    classification, ic_files = run_image_classification(ic_args)
    
    # Check if classification was successful
    if not classification or not ic_files:
        print("Error: Image classification failed.")
        return
    
    # 3. Run VLM Medical Report Generation
    print("\nStep 3: VLM Medical Report Generation")
    
    # Get the paths to the detection and classification result files
    detection_file = next((f for f in od_files if f.endswith('_detection.txt')), None)
    classification_file = next((f for f in ic_files if f.endswith('_classification.txt')), None)
    
    vlm_args = argparse.Namespace(
        image=args.image,
        vlm_model=args.vlm_model,
        detection_file=detection_file,
        classification_file=classification_file,
        save=args.save,
        show=args.show
    )
    
    response, report_path = run_vlm_medical_report(vlm_args)
    
    if report_path:
        print(f"\nFull pipeline completed successfully! Final report: {report_path}")
    else:
        print("\nFull pipeline completed with errors in the VLM stage.")


def main():
    """
    Main function to coordinate the mammography analysis pipeline
    """
    # Parse arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('sample/results', exist_ok=True)
    
    # Check if a mode was specified
    if not args.mode:
        print("Error: No mode specified. Use --help to see available modes.")
        return
    
    # Run the selected mode
    if args.mode == 'od':
        run_object_detection(args)
    elif args.mode == 'ic':
        run_image_classification(args)
    elif args.mode == 'vlm':
        run_vlm_medical_report(args)
    elif args.mode == 'pipeline':
        run_full_pipeline(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()
