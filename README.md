# Mammography Analysis Framework with Object Detection, Classification, and Medical Report Generation

This comprehensive framework provides an end-to-end solution for mammography image analysis using object detection (YOLOv12), image classification (EfficientNetV2B0), and medical report generation (Vision-Language Model). The pipeline offers capabilities from data preprocessing through model training to performance evaluation and medical report generation.

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
│   │   │   ├── train/            # Training images and labels
│   │   │   ├── valid/            # Validation images and labels
│   │   │   ├── test/             # Test images and labels
│   │   │   └── data.yaml         # Dataset configuration
│   │   └── VinDr-Mammo/          # Similar structure as CBIS-DDSM
│   │
│   ├── od_plp/                       # Object Detection module
│   │   ├── pipeline/                 # Implementation modules
│   │   │   ├── CBIS_data_augmentation.py   # CBIS-DDSM data augmentation
│   │   │   ├── VinDr_data_augmentation.py  # VinDr-Mammo data augmentation
│   │   │   ├── VinDr_data_preprocessing.py # Data preprocessing for VinDr
│   │   │   ├── YOLOv8_train.py             # Training script for YOLOv8
│   │   │   ├── YOLOv8_test.py              # Testing and evaluation script
│   │   │   └── YOLOv12_train.py            # Training script for YOLOv12
│   │   ├── od_main.py                # Inference script for object detection
│   │   └── utils/                    # Helper functions and utilities
│   │
│   ├── ic_plp/                       # Image Classification module
│   │   ├── pipeline/                 # Implementation modules
│   │   │   ├── data_preprocessing.py # Data preprocessing for classification
│   │   │   ├── model_train.py        # Training script for classification
│   │   │   └── model_test.py         # Testing and evaluation script
│   │   └── ic_main.py                # Inference script for image classification
│   │
│   ├── vlm_plp/                      # Vision-Language Model module
│   │   ├── vlm_main.py               # Script for medical report generation
│   │   └── utils/                    # Helper functions for VLM
│   │
│   ├── src/                          # Integration scripts
│   │   ├── plp_main.py               # Unified main script integrating OD, IC, and VLM
│   │   └── frontend.py               # Frontend code
│   │   
│   ├── sample/                       # Sample data and results
│   │   ├── sample_data/              # Sample mammography images 
│   │   └── results/                  # Results output directory
│   │
│   └── runs/                         # Training outputs and results
│       ├── detect/                   # Detection model outputs
│       │   ├── YOLOv8_[timestamp]/   # Experiment results of YOLOv8
│       │   │   ├── weights/          # Model checkpoints
│       │   │   └── ...               # TensorBoard logs, metrics, visualizations
│       │   └── YOLOv12/              # Experiment results of YOLOv12
│       │       ├── model_balenced.pt         # Model checkpoints with the most balanced performance
│       │       ├── model_calc_optimized.pt   # Model checkpoints with the best performance on the calc class
│       │       └── model_mass_optimized.pt   # Model checkpoints with the best performance on the mass class
│       └── classification/           # Classification model outputs
│           └── IC_[timestamp]/       # Experiment results
│               ├── weights/          # Model checkpoints
│               └── ...               # Logs and evaluation results
```

## Key Features

- **Multi-Dataset Support**: Works with InBreast, VinDr-Mammo, and CBIS-DDSM datasets
- **YOLOv12 Integration**: Utilizes state-of-the-art YOLOv12 architecture for accurate detection of calcifications and masses
- **Image Classification**: EfficientNetV2B0-based classification of mammograms as benign or malignant
- **Medical Report Generation**: AI-powered generation of comprehensive medical reports using VLM (Vision-Language Model)
- **Comprehensive Data Preprocessing**: Converts various mammography formats to standardized formats
- **Advanced Data Augmentation**: Implements domain-specific augmentations for mammography
- **Fine-Tuned Training**: Customized training pipeline optimized for medical imaging
- **Detailed Evaluation**: Comprehensive metrics including mAP, precision-recall analysis and confusion matrices
- **TensorBoard Integration**: Real-time monitoring of training progress and results visualization
- **Unified Pipeline**: Integrated workflow combining detection, classification, and report generation
- **Flexible API**: Multiple ways to interact with the framework through command line or programmatic API

## Dataset Information

The framework processes three mammography datasets:

- **CBIS-DDSM**: Curated Digital Database for Screening Mammography with verified pathology 
- **VinDr-Mammo**: Large-scale dataset for breast cancer detection from Vietnam
- **InBreast**: High-resolution mammography dataset with detailed annotations

Datasets are preprocessed into standardized format with the following structure:
- `train/images/` - Training images
- `train/labels/` - Labels (normalized bounding box coordinates for detection)
- `valid/images/` - Validation images
- `valid/labels/` - Validation labels
- `data.yaml` - Dataset configuration file

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- TensorFlow 2.9+ (for classification module)
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/mammography-analysis.git
cd mammography-analysis

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Set up environment variables for API keys (for VLM)
# On Windows PowerShell
$env:NIE_QWEN_API_KEY='your_api_key'
# On Windows CMD
set NIE_QWEN_API_KEY=your_api_key
# On macOS/Linux
export NIE_QWEN_API_KEY='your_api_key'
```

## Integrated Pipeline Usage

The framework now provides a unified entry point through `src/main.py` that integrates object detection, image classification, and medical report generation. You can run the complete pipeline or individual components.

### Command Line Interface

#### Complete Pipeline

Run the complete analysis pipeline (detection, classification, and report generation):

```bash
python src/main.py pipeline \
  --image sample/sample_data/sample_image.jpg \
  --od-model runs/detect/YOLOv12/model_balenced.pt \
  --ic-model runs/classification/IC_20250427_170000/weights/best.keras \
  --vlm-model Qwen2.5-VL-7B \
  --od-conf 0.25 \
  --ic-conf 0.5 \
  --save \
  --show
```

#### Object Detection Only

Run only the object detection component:

```bash
python src/main.py od \
  --image sample/sample_data/sample_image.jpg \
  --model runs/detect/YOLOv12/model_balenced.pt \
  --conf 0.25 \
  --save \
  --show
```

#### Image Classification Only

Run only the image classification component:

```bash
python src/main.py ic \
  --image sample/sample_data/sample_image.jpg \
  --model runs/classification/IC_20250427_170000/weights/best.keras \
  --conf 0.5 \
  --imgsz 224 \
  --save \
  --show
```

#### Medical Report Generation Only

Run only the VLM-based medical report generation (requires detection and classification results):

```bash
python src/main.py vlm \
  --image sample/results/sample_image_detection.jpg \
  --model Qwen2.5-VL-7B \
  --detection-file sample/results/sample_image_detection.txt \
  --classification-file sample/results/sample_image_classification.txt \
  --save \
  --show
```

### Common Parameters

- `--image`: Path to the input mammography image
- `--save`: Save results to sample/results/
- `--show`: Display results (requires a GUI environment)

### Mode-Specific Parameters

#### Pipeline Mode Parameters:
- `--od-model`: Path to object detection model weights
- `--ic-model`: Path to image classification model weights
- `--vlm-model`: Name of the VLM model to use
- `--od-conf`: Confidence threshold for object detection
- `--ic-conf`: Confidence threshold for image classification
- `--device`: Device to run on (empty for auto, cpu, 0, 1, etc.)

#### Object Detection Mode Parameters:
- `--model`: Path to object detection model weights
- `--conf`: Confidence threshold for detections
- `--iou`: IoU threshold for NMS
- `--imgsz`: Image size for inference
- `--device`: Device to run on
- `--max-det`: Maximum number of detections per image

#### Image Classification Mode Parameters:
- `--model`: Path to classification model weights
- `--conf`: Confidence threshold for classification
- `--imgsz`: Image size for inference
- `--device`: Device to run on

#### VLM Mode Parameters:
- `--model`: Name of the VLM model to use
- `--detection-file`: Path to detection results file
- `--classification-file`: Path to classification results file
- `--custom-prompt`: Custom prompt to use instead of the default prompt

## Object Detection Module

### Training Guide

The framework uses YOLOv12, a state-of-the-art object detection model, for detecting calcifications and masses in mammography images. Follow these steps to train your own model:

### 1. Prepare Dataset

Ensure your dataset is properly preprocessed and in the YOLOv12 format:
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

Train YOLOv12 model with customized parameters:

```bash
python od_plp/pipeline/YOLOv12_train.py \
  --data dataset/processed_dataset/CBIS-DDSM/data.yaml \
  --model yolo12m.pt \
  --epochs 200 \
  --imgsz 640 \
  --batch 48 \
  --device 0
```

#### Key Training Parameters:

- `--data`: Path to data.yaml file
- `--model`: Base model (yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt)
- `--epochs`: Number of training epochs
- `--imgsz`: Input image size
- `--batch`: Batch size
- `--device`: Computing device (0 for first GPU, cpu for CPU)

### 3. Evaluation

After training, evaluate the model's performance on a test dataset:

```bash
python od_plp/pipeline/YOLOv8_test.py \
  --data dataset/processed_dataset/CBIS-DDSM/data.yaml \
  --model runs/detect/YOLOv12/model_balenced.pt \
  --conf 0.25 \
  --iou 0.7
```

## Image Classification Module

The image classification module uses EfficientNetV2B0 to classify mammograms as benign or malignant.

### Running Classification Inference

```bash
python ic_plp/ic_main.py \
  --image sample/sample_data/sample_image.jpg \
  --model runs/classification/IC_20250427_170000/weights/best.keras \
  --conf 0.5 \
  --imgsz 224 \
  --save \
  --show
```

#### Classification Parameters:
- `--image`: Path to the input mammography image
- `--model`: Path to classification model weights
- `--conf`: Confidence threshold (default: 0.5)
- `--imgsz`: Image size for inference (default: 224)
- `--device`: Device to run on (empty for auto, cpu, 0, 1, etc.)
- `--save`: Save results to sample/results/ directory
- `--show`: Display classification results

#### Example Classification Output:
```
Classification Results for sample_image.jpg:
Class: Benign
Confidence: 52.41%
Benign Probability: 52.41%
Malignant Probability: 47.59%
Inference time: 0.09 seconds
```

## Medical Report Generation Module

The VLM-based medical report generation module uses a Vision-Language Model to generate comprehensive medical reports based on the image, detection results, and classification results.

### Running VLM Inference

```bash
python vlm_plp/vlm_main.py \
  --image sample/results/sample_image_detection.jpg \
  --model Qwen2.5-VL-7B \
  --detection-file sample/results/sample_image_detection.txt \
  --classification-file sample/results/sample_image_classification.txt \
  --save \
  --show
```

#### VLM Parameters:
- `--image`: Path to the input mammography image
- `--model`: Name of the VLM model to use (default: Qwen2.5-VL-7B)
- `--detection-file`: Path to detection results file
- `--classification-file`: Path to classification results file
- `--save`: Save results to sample/results/ directory
- `--show`: Display the generated report

#### Example VLM Output:
The VLM generates a comprehensive medical report with two main sections:
1. **Data Analysis Summary**: Technical analysis of detection and classification results
2. **Professional Radiological Assessment**: Expert-like interpretation with medical terminology, including BI-RADS assessment and follow-up recommendations

## Streamlit Web Interface

The framework includes a user-friendly web interface built with Streamlit, allowing users to interact with the mammography analysis platform through a browser-based UI.

### Features

- **Interactive UI**: Modern, clean interface for uploading and analyzing mammography images
- **Real-time Analysis**: Process mammography images and view results in real-time
- **Comprehensive Results Display**: Organized tabs for object detection, classification, and medical report results
- **Download Options**: Easily download detection results, classification details, and medical reports
- **Progress Tracking**: Visual progress indicators during analysis pipeline execution

### Running the Web Interface

```bash
# Navigate to the src directory
cd src

# Launch the Streamlit app
streamlit run frontend.py
```

The interface will be accessible in your web browser at `http://localhost:8501`.

### Using the Web Interface

1. **Upload Image**: Click the file uploader to select a mammography image (JPG, JPEG, or PNG)
2. **Generate Analysis**: Click the "Generate Comprehensive Analysis" button
3. **View Results**: Explore the results in three tabs:
   - **Object Detection**: View detected abnormalities with bounding boxes and detection details
   - **Classification**: See benign/malignant classification results with confidence scores
   - **Medical Report**: Read the comprehensive AI-generated medical report
4. **Download Results**: Use the download buttons to save detection results, classification details, and the medical report

## Performance Metrics

The framework evaluates models using several metrics:

- **Object Detection**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Image Classification**: Accuracy, Precision, Recall, F1-Score, AUC
- **Medical Report Generation**: Qualitative assessment by medical professionals

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**:
   - Ensure `NIE_QWEN_API_KEY` is set for VLM functionality

2. **Model Loading Issues**:
   - Verify model paths are correct
   - For classification models, ensure TensorFlow is properly installed

3. **API Connection Problems**:
   - Check internet connection for VLM API calls
   - Verify API key is valid and has sufficient permissions

4. **Integration Pipeline Errors**:
   - Make sure all component modules are functioning independently first
   - Check that output files from earlier stages exist before running later stages

5. **CUDA Out of Memory**:
   - Reduce batch size
   - Try a smaller model variant

## Acknowledgements

- YOLOv12 implementation by [Ultralytics](https://github.com/ultralytics/ultralytics)
- The CBIS-DDSM dataset by [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- VinDr-Mammo dataset by [Vingroup Big Data Institute](https://vindr.ai/datasets/mammo)
- InBreast dataset by [INESC Porto](https://www.inf.ufg.br/~rogerio/inbreast/)
- Vision-Language Model provided by NIE 