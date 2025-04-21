# Object Detection Framework for Mammography

This project is a framework for detecting calcifications (calc) and masses in mammography images. It provides an end-to-end pipeline for preprocessing data, training models, and evaluating detection performance.

## Repository Structure

```
.
├── dataset/
│   ├── original dataset/
│   │   ├── InBreast/             # InBreast mammography dataset
│   │   ├── VinDr-Mammo/          # VinDr-Mammo dataset
│   │   └── CBIS-DDSM/            # CBIS-DDSM dataset
│   └── processed dataset/        # Stores preprocessed datasets
│
├── od_plp/                       # Main code directory
│   ├── config/                   # Configuration parameters
│   ├── pipeline/                 # Implementation steps
│   │   ├── data_preprocessing/   # Data preprocessing modules
│   │   ├── model_initialization/ # Model architecture and setup
│   │   ├── model_train/          # Training procedures
│   │   ├── model_test/           # Testing procedures
│   │   └── model_evaluate/       # Evaluation metrics and visualization
│   ├── utils/                    # Helper functions and utilities
│   └── main.py                   # Main execution script
│
└── results/                      # TensorBoard logs and model outputs
```

## Dataset Information

The framework works with multiple mammography datasets:

- **InBreast**: High-resolution mammography dataset with accurate annotations
- **VinDr-Mammo**: Large-scale dataset for breast cancer detection
- **CBIS-DDSM**: Comprehensive collection of digitized film mammograms with verified pathology information

Original images are stored in the `original dataset` directory, while preprocessed data ready for model training is saved to the `processed dataset` directory.

## Framework Components

### Configuration (od_plp/config/)

The configuration directory contains parameters for:
- Model hyperparameters
- Training settings (batch size, learning rate, epochs)
- Dataset paths
- Augmentation settings
- Detection thresholds

### Pipeline (od_plp/pipeline/)

The pipeline is divided into distinct modules:

1. **Data Preprocessing**
   - Image normalization
   - Annotation conversion
   - Dataset splitting
   - Data augmentation

2. **Model Initialization**
   - Model architecture selection
   - Backbone networks
   - Detection heads
   - Parameter initialization

3. **Model Training**
   - Training loop implementation
   - Loss functions
   - Optimizer configuration
   - Checkpoint saving

4. **Model Testing**
   - Inference pipeline
   - Batch processing
   - Detection post-processing

5. **Model Evaluation**
   - Precision-Recall calculation
   - mAP (mean Average Precision)
   - Visualization tools
   - Confusion matrices

### Utilities (od_plp/utils/)

Helper functions for:
- Data loading and manipulation
- Visualization
- Metric calculation
- File I/O operations
- Logging

### Main Execution (od_plp/main.py)

The main script provides a complete workflow:
- **Input**: Mammography images
- **Output**: Detection results with labels (calc/mass) and bounding boxes

## Result Tracking

The `results` directory uses TensorBoard to:
- Monitor training progress in real-time
- Record evaluation metrics
- Visualize detection examples
- Compare different model configurations

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/username/mammography-detection.git
cd mammography-detection

# Install dependencies
pip install -r requirements.txt

# Set up datasets
mkdir -p dataset/original\ dataset
# Download and place datasets in the original dataset directory
```

### Usage

#### Data Preprocessing

```bash
# Preprocess the datasets
python od_plp/pipeline/data_preprocessing/preprocess.py
```

#### Training

```bash
# Train the model
python od_plp/main.py --mode train
```

#### Testing and Evaluation

```bash
# Test the model
python od_plp/main.py --mode test

# Evaluate the model
python od_plp/main.py --mode evaluate
```

#### Visualization

```bash
# Launch TensorBoard to view results
tensorboard --logdir=results
```

## Configuration

You can modify model parameters and training settings by editing the configuration files in the `od_plp/config/` directory:

```bash
# Example: change the training configuration
vi od_plp/config/train_config.py
```

## License

[Specify license information here]

## Acknowledgements

- List datasets used with appropriate citations
- Reference any base models or architectures adopted
- Acknowledge any other resources or inspirations 