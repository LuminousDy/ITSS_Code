import os
import argparse
from ultralytics import YOLO
from datetime import datetime


def parse_args():
    """
    Parse command line arguments for YOLOv8 training
    """
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for mammography object detection')
    parser.add_argument('--data', type=str, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=48, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device to run on (cpu, 0, 1, 2, 3, etc.)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads for data loading')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model to train, e.g. yolov8n.pt, yolov8s.pt, etc.')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project name')
    parser.add_argument('--name', type=str, default=None, help='Experiment name (default: YOLOv8_timestamp)')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every x epochs')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate (fraction of lr0)')
    
    return parser.parse_args()


def train_yolov8(args):
    """
    Train a YOLOv8 model with the specified parameters
    """
    # Load a model
    model = YOLO(args.model)  # load a pretrained model
    
    # Create experiment name with timestamp if not provided
    if args.name is None:
        args.name = f"YOLOv8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Convert data path to absolute path if it's not already
    if args.data and not os.path.isabs(args.data):
        args.data = os.path.abspath(args.data)
        print(f"Using absolute data path: {args.data}")
    
    # Verify that the data.yaml file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    print(f"Training YOLOv8 model with the following parameters:")
    print(f"- Data: {args.data}")
    print(f"- Model: {args.model}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Image size: {args.imgsz}")
    print(f"- Batch size: {args.batch}")
    print(f"- Device: {args.device if args.device else 'auto'}")
    print(f"- Project: {args.project}")
    print(f"- Experiment name: {args.name}")
    
    # Print Ultralytics settings for debugging
    settings_path = os.path.expanduser("~/.config/Ultralytics/settings.yaml")
    if os.name == 'nt':  # Windows
        settings_path = os.path.join(os.getenv('APPDATA'), 'Ultralytics', 'settings.json')
    
    print(f"Ultralytics settings path: {settings_path}")
    if os.path.exists(settings_path):
        print(f"Settings file exists: {os.path.exists(settings_path)}")
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        lr0=args.lr0,
        lrf=args.lrf,
        exist_ok=True,       # Overwrite existing experiment
        pretrained=True,     # Start from pretrained model
        optimizer='AdamW',     # Optimizer (choices: SGD, Adam, AdamW, etc.)
        warmup_epochs=3,
        plots=True,          # Save plots during training
        verbose=True         # Print verbose output
    )
    
    # Return the results
    return model, results


def evaluate_model(model, args):
    """
    Evaluate the trained model on the validation set
    """
    print("\nEvaluating model on validation set...")
    metrics = model.val(data=args.data)
    
    # Print metrics
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.5f}")
    print(f"mAP50-95: {metrics.box.map:.5f}")
    
    return metrics


def main():
    """
    Main function to run YOLOv8 training
    """
    # Parse command line arguments
    args = parse_args()
    
    # Train the model
    model, results = train_yolov8(args)
    
    # Evaluate the model
    metrics = evaluate_model(model, args)
    
    # Print info about where to find TensorBoard logs
    log_dir = os.path.join(args.project, args.name)
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {log_dir}")
    print(f"TensorBoard logs are available. To view them, run:")
    print(f"tensorboard --logdir={log_dir}")
    
    return model, results, metrics


if __name__ == "__main__":
    model, results, metrics = main()
