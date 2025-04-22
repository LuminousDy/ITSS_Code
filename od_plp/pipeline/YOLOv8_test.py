import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import seaborn as sns


def parse_args():
    """
    Parse command line arguments for YOLOv8 evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model for mammography object detection')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights (*.pt)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for testing')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device to run on (cpu, 0, 1, 2, 3, etc.)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum detections per image')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project name')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='Save results to *.json')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--save-hybrid', action='store_true', help='Save hybrid version of labels (labels + additional predictions)')
    
    return parser.parse_args()


def evaluate_model(args):
    """
    Evaluate a YOLOv8 model on a test dataset
    """
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Create experiment name with timestamp if not provided
    if args.name is None:
        args.name = f"eval_{Path(args.model).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Convert to absolute path if needed
    if args.data and not os.path.isabs(args.data):
        args.data = os.path.abspath(args.data)
    
    print(f"Evaluating model on {args.data}...")
    print(f"- Model: {args.model}")
    print(f"- Image size: {args.imgsz}")
    print(f"- Batch size: {args.batch}")
    print(f"- Confidence threshold: {args.conf}")
    print(f"- IoU threshold: {args.iou}")
    
    # Run validation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_json=args.save_json,
        verbose=args.verbose,
        save_hybrid=args.save_hybrid,
        plots=True  # Generate plots
    )
    
    return model, metrics


def display_metrics(metrics):
    """
    Display metrics from model evaluation
    """
    # Extract metrics
    mp = metrics.box.mp  # mean precision
    mr = metrics.box.mr  # mean recall
    map50 = metrics.box.map50  # mAP at IoU 0.5
    map = metrics.box.map  # mAP at IoU 0.5-0.95
    
    # Per-class metrics
    precision = metrics.box.p  # precision per class
    recall = metrics.box.r  # recall per class
    ap50 = metrics.box.ap50  # AP at IoU 0.5 per class
    ap = metrics.box.ap  # AP at IoU 0.5-0.95 per class
    
    # Get class names if available
    if hasattr(metrics, 'names'):
        # Convert class_names to list if it's a dict
        if isinstance(metrics.names, dict):
            # Sort by key to maintain consistent order
            max_key = max(metrics.names.keys())
            class_names = [metrics.names.get(i, f'Class {i}') for i in range(max_key + 1)]
        else:
            class_names = list(metrics.names)
    else:
        class_names = [f'Class {i}' for i in range(len(precision))]
    
    # Print overall metrics
    print("\n" + "="*50)
    print("Overall Performance Metrics:")
    print(f"Mean Precision: {mp:.4f}")
    print(f"Mean Recall: {mr:.4f}")
    print(f"mAP@0.5: {map50:.4f}       (Average Precision at IoU threshold 0.5)")
    print(f"mAP@0.5:0.95: {map:.4f}   (Average Precision across IoU thresholds 0.5-0.95)")
    print("="*50)
    
    # Print IoU explanation
    print("\nIoU (Intersection over Union) Explanation:")
    print("IoU measures how well the predicted boxes overlap with ground truth boxes.")
    print("- mAP@0.5: Mean Average Precision where a detection is considered correct if IoU ≥ 0.5")
    print("- mAP@0.5:0.95: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95")
    print("="*50)
    
    # Print per-class metrics
    print("\nPer-class Performance:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'AP@0.5':<10} {'AP@0.5:0.95':<10}")
    print("-"*60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {precision[i]:.4f}      {recall[i]:.4f}      {ap50[i]:.4f}      {ap[i]:.4f}")
    
    # Return a dictionary of metrics for further analysis
    metric_dict = {
        'mp': mp,
        'mr': mr,
        'map50': map50,
        'map': map,
        'precision': precision,
        'recall': recall,
        'ap50': ap50,
        'ap': ap,
        'class_names': class_names
    }
    
    return metric_dict


def create_additional_plots(metrics, save_dir):
    """
    Create additional plots beyond what YOLOv8 provides by default
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract per-class metrics
    precision = metrics.box.p
    recall = metrics.box.r
    ap50 = metrics.box.ap50
    ap = metrics.box.ap
    
    # Get class names if available
    if hasattr(metrics, 'names'):
        # Convert class_names to list if it's a dict
        if isinstance(metrics.names, dict):
            # Sort by key to maintain consistent order
            max_key = max(metrics.names.keys())
            class_names = [metrics.names.get(i, f'Class {i}') for i in range(max_key + 1)]
        else:
            class_names = list(metrics.names)
    else:
        class_names = [f'Class {i}' for i in range(len(precision))]
    
    # Ensure all data are lists and of the same length
    n_classes = len(precision)
    if len(class_names) < n_classes:
        # Extend class_names if needed
        class_names.extend([f'Class {i}' for i in range(len(class_names), n_classes)])
    elif len(class_names) > n_classes:
        # Truncate class_names if needed
        class_names = class_names[:n_classes]
    
    # Convert numpy arrays to lists
    precision_list = precision.tolist() if hasattr(precision, 'tolist') else list(precision)
    recall_list = recall.tolist() if hasattr(recall, 'tolist') else list(recall)
    ap50_list = ap50.tolist() if hasattr(ap50, 'tolist') else list(ap50)
    ap_list = ap.tolist() if hasattr(ap, 'tolist') else list(ap)
    
    # Create a list of dictionaries for DataFrame
    data = []
    for i in range(n_classes):
        data.append({
            'Class': class_names[i],
            'Precision': precision_list[i],
            'Recall': recall_list[i],
            'AP@0.5': ap50_list[i],
            'AP@0.5:0.95': ap_list[i]
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print the DataFrame to debug
    print("\nData for plots:")
    print(df)
    
    try:
        # 1. Bar plot of precision and recall
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(class_names))
        
        plt.bar(index, df['Precision'], bar_width, label='Precision')
        plt.bar(index + bar_width, df['Recall'], bar_width, label='Recall')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Precision and Recall by Class')
        plt.xticks(index + bar_width/2, df['Class'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_by_class.png'))
        plt.close()
        
        # 2. Heatmap of metrics
        plt.figure(figsize=(10, 8))
        heatmap_data = df[['Precision', 'Recall', 'AP@0.5', 'AP@0.5:0.95']].values
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', 
                    xticklabels=['Precision', 'Recall', 'AP@0.5', 'AP@0.5:0.95'],
                    yticklabels=df['Class'])
        plt.title('Performance Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'))
        plt.close()
        
        # 3. Confusion matrix if available
        if hasattr(metrics, 'confusion_matrix'):
            try:
                cm = metrics.confusion_matrix
                plt.figure(figsize=(10, 8))
                # Add background class if needed
                labels = list(df['Class'])
                if cm.matrix.shape[0] > len(labels):
                    labels = labels + ['background']
                
                sns.heatmap(cm.matrix, annot=True, fmt='.2f', cmap='Blues',
                            xticklabels=labels,
                            yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
                plt.close()
            except Exception as e:
                print(f"Error creating confusion matrix plot: {e}")
        
        print(f"Additional plots saved to {save_dir}")
    except Exception as e:
        print(f"Error creating plots: {e}")
        # Generate at least one simple plot to avoid complete failure
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(df['Class'], df['AP@0.5'])
            plt.xlabel('Class')
            plt.ylabel('AP@0.5')
            plt.title('Average Precision (IoU=0.5) by Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'ap50_by_class.png'))
            plt.close()
            print(f"Fallback plot saved to {save_dir}")
        except Exception as e2:
            print(f"Even fallback plot failed: {e2}")


def run_predictions(model, args):
    """
    Run predictions on a test set and save the results
    """
    print("\nRunning predictions on test set...")
    
    # Parse data.yaml to find test images path
    import yaml
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Get the base directory of the data configuration file
    data_yaml_dir = os.path.dirname(os.path.abspath(args.data))
    
    # Try to determine the test dataset directory
    # First, try using the test path
    test_dir = data_cfg.get('test', None)
    test_exists = False
    
    if test_dir:
        # Handle relative paths
        abs_test_dir = resolve_path(data_yaml_dir, test_dir)
        if os.path.exists(abs_test_dir):
            test_exists = True
            test_dir = abs_test_dir
    
    # If the test directory does not exist, try using the validation set
    if not test_exists:
        print(f"Test directory not found or doesn't exist. Trying validation set instead.")
        val_dir = data_cfg.get('val', None)
        
        if not val_dir:
            # Try other common validation set directory names
            for alt_name in ['valid', 'validation']:
                if alt_name in data_cfg:
                    val_dir = data_cfg[alt_name]
                    break
        
        if val_dir:
            abs_val_dir = resolve_path(data_yaml_dir, val_dir)
            if os.path.exists(abs_val_dir):
                print(f"Using validation directory for predictions: {abs_val_dir}")
                test_dir = abs_val_dir
            else:
                # Special handling: try to infer the validation set path directly from base_dir
                # This is a special case for CBIS-DDSM
                dataset_base_dir = os.path.dirname(data_yaml_dir)
                possible_valid_dir = os.path.join(dataset_base_dir, 'valid', 'images')
                if os.path.exists(possible_valid_dir):
                    print(f"Found validation directory by inference: {possible_valid_dir}")
                    test_dir = possible_valid_dir
                else:
                    # Try to find it at the same level as data.yaml
                    possible_valid_dir = os.path.join(data_yaml_dir, 'valid', 'images')
                    if os.path.exists(possible_valid_dir):
                        print(f"Found validation directory at same level: {possible_valid_dir}")
                        test_dir = possible_valid_dir
                    else:
                        raise FileNotFoundError(f"Neither test nor validation directories could be found")
        else:
            raise ValueError("Neither test nor validation paths defined in data.yaml")
    
    print(f"Running predictions on: {test_dir}")
    
    try:
        # Run predictions
        results = model.predict(
            source=test_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            project=args.project,
            name=f"{args.name}_pred",
            save=True,  # Save images with predictions
            save_txt=args.save_txt,  # Save results to *.txt
            save_conf=args.save_conf,  # Save confidences in --save-txt labels
            verbose=args.verbose
        )
        
        return results
    except Exception as e:
        print(f"Error running predictions: {e}")
        print("Skipping prediction step.")
        return None


def resolve_path(base_dir, rel_path):
    """
    Resolve relative paths, handling special cases like ../
    """
    # Normalize path separators to the system's separator
    rel_path = rel_path.replace('/', os.path.sep).replace('\\', os.path.sep)
    
    # If it's an absolute path, return it directly
    if os.path.isabs(rel_path):
        return rel_path
    
    # If it contains relative path symbols, use os.path.normpath to resolve
    if '..' in rel_path:
        # On Windows, os.path.join may have issues with paths containing ..
        # So we combine first, then normalize
        combined_path = os.path.join(base_dir, rel_path)
        return os.path.normpath(combined_path)
    else:
        # Simple relative path, just concatenate
        return os.path.join(base_dir, rel_path)


def main():
    """
    Main function to run YOLOv8 evaluation
    """
    # Parse command line arguments
    args = parse_args()
    
    # Evaluate model
    model, metrics = evaluate_model(args)
    
    # Display results
    metric_dict = display_metrics(metrics)
    
    # Create additional plots
    save_dir = os.path.join(args.project, args.name)
    create_additional_plots(metrics, save_dir)
    
    # Run predictions to generate visualizations
    results = run_predictions(model, args)
    
    print(f"\nEvaluation completed. Results saved to: {save_dir}")
    
    return model, metrics, results


if __name__ == "__main__":
    model, metrics, results = main()
