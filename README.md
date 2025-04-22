# Mammography Object Detection Framework with YOLOv8

This framework provides a comprehensive solution for detecting calcifications and masses in mammography images using YOLOv8. The pipeline offers end-to-end capabilities from data preprocessing through model training to performance evaluation.

## Repository Structure

```
.
├── dataset/
│   ├── original dataset/
│   │   ├── InBreast/             # InBreast mammography dataset
│   │   ├── VinDr-Mammo/          # VinDr-Mammo dataset
│   │   └── CBIS-DDSM/            # CBIS-DDSM dataset
│   └── processed dataset/        # Preprocessed datasets in YOLOv8 format
│       ├── CBIS-DDSM/
│       │   ├── train/            # Training images and labels
│       │   ├── valid/            # Validation images and labels
│       │   ├── test/             # Test images and labels
│       │   └── data.yaml         # Dataset configuration
│       └── VinDr-Mammo/          # Similar structure as CBIS-DDSM
│
├── od_plp/                       # Main code directory
│   ├── pipeline/                 # Implementation modules
│   │   ├── CBIS_data_augmentation.py   # CBIS-DDSM data augmentation
│   │   ├── VinDr_data_augmentation.py  # VinDr-Mammo data augmentation
│   │   ├── VinDr_data_preprocessing.py # Data preprocessing for VinDr
│   │   ├── YOLOv8_train.py             # Training script for YOLOv8
│   │   └── YOLOv8_test.py              # Testing and evaluation script
│   ├── main.py                   # Main inference script for single image detection
│   └── utils/                    # Helper functions and utilities
│
├── sample/                       # Sample data and results
│   ├── sample_data/              # Sample mammography images 
│   └── results/                  # Detection results output
│
└── runs/                         # Training outputs and results
    └── detect/                   # Detection results
        └── YOLOv8_[timestamp]/   # Experiment results
            ├── weights/          # Model checkpoints
            └── ...               # TensorBoard logs, metrics, visualizations
```

## Key Features

- **Multi-Dataset Support**: Works with InBreast, VinDr-Mammo, and CBIS-DDSM datasets
- **YOLOv8 Integration**: Utilizes state-of-the-art YOLOv8 architecture for accurate detection
- **Comprehensive Data Preprocessing**: Converts various mammography formats to standardized YOLOv8 format
- **Advanced Data Augmentation**: Implements domain-specific augmentations for mammography
- **Fine-Tuned Training**: Customized training pipeline optimized for medical imaging
- **Detailed Evaluation**: Comprehensive metrics including mAP, precision-recall analysis and confusion matrices
- **TensorBoard Integration**: Real-time monitoring of training progress and results visualization
- **Single Image Inference**: Detect calcifications and masses in individual mammography images

## Dataset Information

The framework processes three mammography datasets:

- **CBIS-DDSM**: Curated Digital Database for Screening Mammography with verified pathology 
- **VinDr-Mammo**: Large-scale dataset for breast cancer detection from Vietnam
- **InBreast**: High-resolution mammography dataset with detailed annotations

Datasets are preprocessed into standardized YOLOv8 format with the following structure:
- `train/images/` - Training images
- `train/labels/` - YOLOv8 format labels (normalized bounding box coordinates)
- `valid/images/` - Validation images
- `valid/labels/` - Validation labels
- `data.yaml` - Dataset configuration file

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/mammography-detection.git
cd mammography-detection

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install required packages
pip install ultralytics pandas numpy matplotlib seaborn tqdm albumentations opencv-python

# Set up datasets
# Place original datasets in dataset/original dataset/[DATASET_NAME]
```

## Complete Pipeline Workflow

### 1. Data Preprocessing

Convert mammography datasets to YOLOv8 format:

```bash
# Process VinDr-Mammo dataset
python od_plp/pipeline/VinDr_data_preprocessing.py

# Process CBIS-DDSM dataset (if not already in YOLOv8 format)
# Note: CBIS-DDSM often comes pre-processed
```

### 2. Data Augmentation

Apply specific augmentations to improve model robustness:

```bash
# Augment VinDr-Mammo dataset
python od_plp/pipeline/VinDr_data_augmentation.py

# Augment CBIS-DDSM dataset
python od_plp/pipeline/CBIS_data_augmentation.py
```

#### Implemented Augmentations:

- **Image Preprocessing**:
  - Resize to 416×416
  - Auto-contrast via adaptive equalization (CLAHE)
  
- **Augmentation Techniques**:
  - Random crop (0-20%)
  - Salt and pepper noise (5% of pixels)
  - Random rotation (±15°)
  - Random horizontal flip
  - Brightness and contrast adjustments
  - Gaussian blur

## Training Guide

The framework uses YOLOv8, a state-of-the-art object detection model, for detecting calcifications and masses in mammography images. Follow these steps to train your own model:

### 1. Prepare Dataset

Ensure your dataset is properly preprocessed and in the YOLOv8 format:
- Images in the correct directories (`train/images/`, `valid/images/`)
- Labels in the correct format and directories (`train/labels/`, `valid/labels/`)
- `data.yaml` file with proper configuration

The `data.yaml` file should have the following structure:
```yaml
train: path/to/train/images
val: path/to/valid/images
test: path/to/test/images  # Optional

nc: 2  # Number of classes
names: ['Mass', 'Calc']  # Class names
```

### 2. Start Training

Train YOLOv8 model with customized parameters:

```bash
python od_plp/pipeline/YOLOv8_train.py \
  --data dataset/processed_dataset/CBIS-DDSM/data.yaml \
  --model yolov8s.pt \
  --epochs 200 \
  --imgsz 640 \
  --batch 48 \
  --device 0
```

#### Key Training Parameters:

- `--data`: Path to data.yaml file
- `--model`: Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `--epochs`: Number of training epochs
- `--imgsz`: Input image size
- `--batch`: Batch size
- `--device`: Computing device (0 for first GPU, cpu for CPU)

#### Advanced Training Parameters:

- `--patience`: Early stopping patience (default: 50)
- `--save-period`: Save checkpoint every x epochs (default: 10)
- `--lr0`: Initial learning rate (default: 0.01)
- `--lrf`: Final learning rate as a fraction of lr0 (default: 0.01)
- `--workers`: Number of worker threads for data loading (default: 8)
- `--project`: Project directory (default: runs/detect)
- `--name`: Experiment name (default: YOLOv8_[timestamp])

### 3. Monitor Training

The training script automatically logs training progress and saves checkpoints. You can monitor the training using TensorBoard:

```bash
tensorboard --logdir=runs/detect/YOLOv8_[timestamp]
```

Training metrics, model performance, and sample detections will be displayed in the TensorBoard dashboard.

### 4. Training Output

After training is complete, the following outputs will be generated:
- **Metrics**: Precision, recall, mAP@0.5, mAP@0.5:0.95
- **Visualizations**: PR curve, confusion matrix, examples of predictions
- **Model weights**:
  - `best.pt`: Best model weights according to validation mAP
  - `last.pt`: Latest model weights
  - `epoch_N.pt`: Model weights at specific epochs (based on save-period)

## Evaluation Guide

After training, evaluate the model's performance on a test dataset:

```bash
python od_plp/pipeline/YOLOv8_test.py \
  --data dataset/processed_dataset/CBIS-DDSM/data.yaml \
  --model runs/detect/YOLOv8_20250422_122835/weights/best.pt \
  --conf 0.25 \
  --iou 0.7
```

#### Evaluation Parameters:

- `--data`: Path to data.yaml file
- `--model`: Path to trained model weights
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)
- `--save-txt`: Save results to text files
- `--save-json`: Save results in COCO JSON format
- `--verbose`: Print verbose output

The evaluation script will:
1. Load the specified model
2. Evaluate it on the validation or test dataset
3. Calculate and display performance metrics
4. Generate visualizations (confusion matrix, PR curves, etc.)
5. Run predictions on test images and save the results

## Inference Guide

The framework provides two ways to perform inference on single mammography images:

### 1. Command Line Interface

Use the `main.py` script to detect calcifications and masses in a single image:

```bash
python od_plp/main.py \
  --image sample/sample_data/sample_image.jpg \
  --conf 0.25 \
  --save \
  --show
```

#### Required Parameters:
- `--image`: Path to the input mammography image

#### Optional Parameters:
- `--model`: Path to model weights (default: runs/detect/YOLOv8_20250422_122835/weights/best.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)
- `--imgsz`: Image size for inference (default: 640)
- `--device`: Device to run on (empty for auto, cpu, 0, 1, etc.)
- `--save`: Save results to sample/results/ directory
- `--show`: Display detection results (requires a GUI environment)
- `--max-det`: Maximum number of detections per image (default: 300)

#### Example Output:
```
Loading model from runs/detect/YOLOv8_20250422_122835/weights/best.pt...
Running detection on sample/sample_data/sample_image.jpg...

Detection Results for sample_image.jpg:
Found 2 objects
  1. Mass: 0.5162, Box: [197, 248, 225, 260]
  2. Calc: 0.3419, Box: [221, 259, 230, 264]
Inference completed in 0.99 seconds
Results saved to sample/results
  - Detection image: sample/results/sample_image_detection.jpg
  - Detection data: sample/results/sample_image_detection.txt
```

### 2. Programmatic API

Import and use the `infer_single_image` function in your Python code:

```python
from od_plp.main import infer_single_image

# Perform detection
detections = infer_single_image(
    image_path="sample/sample_data/sample_image.jpg",
    model_path="runs/detect/YOLOv8_20250422_122835/weights/best.pt",
    conf=0.25,
    show=True,
    save=True
)

# Process detection results
for i, det in enumerate(detections):
    print(f"Detection {i+1}:")
    print(f"  Class: {det['class']}")
    print(f"  Confidence: {det['confidence']:.4f}")
    print(f"  Bounding Box: {det['bbox']}")
```

#### Parameters:
- `image_path`: Path to the input mammography image
- `model_path`: Path to model weights (optional)
- `conf`: Confidence threshold (default: 0.25)
- `show`: Whether to display results (default: False)
- `save`: Whether to save results (default: False)

#### Return Value:
A list of dictionaries, each containing:
- `class`: Detected class name (Mass or Calc)
- `confidence`: Detection confidence score
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]

### 3. Interpreting Detection Results

The detection results include:
- **Class**: "Mass" or "Calc" (calcification)
- **Confidence**: A score between 0 and 1 indicating the model's confidence
- **Bounding Box**: The coordinates [x1, y1, x2, y2] defining the location of the detected abnormality

The saved output includes:
1. An annotated image with colored bounding boxes (red for Mass, green for Calc)
2. A text file with detailed detection information

## Performance Metrics

The framework evaluates models using several metrics:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds
- **Precision**: How many of the detected objects are relevant
- **Recall**: How many relevant objects are detected
- **IoU (Intersection over Union)**: Measures bounding box accuracy

## Troubleshooting

### Common Issues

1. **Missing Dataset Files**:
   - Ensure all dataset directories follow the expected structure
   - Check that data.yaml correctly points to image directories

2. **CUDA Out of Memory**:
   - Reduce batch size
   - Try a smaller YOLOv8 model variant

3. **Path Errors in Windows**:
   - Use double backslashes or raw strings for Windows paths
   - Alternatively, use forward slashes which work cross-platform

4. **TensorBoard Not Showing Data**:
   - Verify TensorBoard is enabled in Ultralytics settings
   - Check log directory path is correct

5. **Inference Issues**:
   - Ensure the model weights file exists at the specified path
   - Try lowering the confidence threshold if no detections are found
   - Check that input images are in a supported format (jpg, png, etc.)

## Acknowledgements

- YOLOv8 implementation by [Ultralytics](https://github.com/ultralytics/ultralytics)
- The CBIS-DDSM dataset by [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- VinDr-Mammo dataset by [Vingroup Big Data Institute](https://vindr.ai/datasets/mammo)
- InBreast dataset by [INESC Porto](https://www.inf.ufg.br/~rogerio/inbreast/) 