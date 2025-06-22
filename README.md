# MNIST Handwritten Digit Classification

This project implements a complete training pipeline for MNIST handwritten digit classification using PyTorch. The system trains different CNN architectures and provides comprehensive analysis tools with an interactive menu system.

## Features

- **Three CNN Models**: 1-layer, 2-layer, and 3-layer convolutional neural networks
- **Data Augmentation**: Configurable augmentation strategies for improved generalization
- **Model Training**: Complete training pipeline with validation and early stopping
- **Best Model Saving**: Automatic saving of best models with descriptive filenames including timestamps and accuracy
- **Comprehensive Analysis**: Training history plots, confusion matrices with counts and percentages
- **Interactive Menu System**: Easy-to-use interface for different operations
- **Drawn Digit Testing**: Test trained models with hand-drawn digits and generate saliency maps
- **Sample Data Visualization**: Display and analyze sample MNIST data
- **Robust Model Loading**: Handles different model architectures and configurations

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- PIL (Pillow)
- numpy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mnist_del2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data drawn_digits outputs
```

## Usage

### Running the Main Program

```bash
python main.py
```

The program provides an interactive menu with the following options:

1. **Train all models** - Trains all CNN models with data augmentation and comparison studies
2. **Display sample data** - Shows sample MNIST digits and dataset statistics
3. **Test best model with drawn images** - Tests the best trained model with hand-drawn digits
4. **Draw and test a digit** - Interactive drawing and testing workflow (draw a digit and test it immediately)
5. **Exit** - Exits the program

### Workflow

1. **First, train the models** by selecting option 1
2. **Draw digits for testing** using the drawing utility
3. **Test the models** by selecting option 3 or 4
   - Option 3: Test with existing drawn images
   - Option 4: Draw and test interactively (recommended for quick testing)
4. **View results** in the outputs directory

### Drawing Digits for Testing

To test the models with your own hand-drawn digits:

1. Run the drawing utility:
```bash
python draw_digit.py
```

2. Draw a digit in the popup window
3. Save the digit (it will be saved as `drawn_digits/drawn_digit.png`)
4. Use option 3 in the main menu to test the digit

## Model Architectures

### Model 1 (1-Layer CNN)
- 1 convolutional layer (32 filters)
- Max pooling
- 2 fully connected layers
- No batch normalization or dropout
- **Training**: With data augmentation

### Model 2 (2-Layer CNN)
- 2 convolutional layers (32, 64 filters)
- Max pooling after each conv layer
- 2 fully connected layers
- Batch normalization and dropout (0.2)
- **Training**: With data augmentation

### Model 3 (3-Layer CNN) - Comparison Study
- 3 convolutional layers (32, 64, 128 filters)
- Max pooling after each conv layer
- 2 fully connected layers
- Batch normalization and dropout (0.3)
- **Trained twice**: once with data augmentation, once without (for comparison)

## Data Augmentation

The training pipeline includes configurable data augmentation:
- Random rotation (±10 degrees)
- Random affine transformations (translation and scaling)
- Random erasing (configurable probability)
- Random cropping (optional)

**Important**: Augmentation is only applied to training data. Validation and test sets use basic transforms to ensure fair evaluation.

## Output Structure

All outputs are saved in timestamped directories under `outputs/`:

```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── models/          # Saved model checkpoints
    ├── plots/           # Training plots and confusion matrices
    └── logs/            # Training logs
```

### Model Files
- Best models: `{model_name}_best_{timestamp}_acc_{accuracy}.pth`
- Epoch checkpoints: `{model_name}_epoch_{epoch}.pth`

### Plot Files
- Training history: `{model_name}_training_history.png`
- Confusion matrices: `{model_name}_confusion_matrix.png`
- Sample data: `sample_data.png`
- Drawn digit analysis: `drawn_digit_analysis.png`

## Key Features

### Best Model Saving
Models are automatically saved with descriptive names including:
- Model architecture (1layer, 2layer, 3layer_aug, 3layer_noaug)
- Timestamp
- Validation accuracy
- Complete model state and parameters

### Enhanced Confusion Matrices
Confusion matrices show both:
- Raw counts for each class
- Percentages for better interpretability

### Saliency Maps
When testing with drawn digits, the system generates saliency maps showing which parts of the image the model focuses on for classification.

### Robust Model Loading
The system can handle loading models with different architectures:
- Automatically detects model type from filename
- Recreates model with correct parameters (batch norm, dropout, etc.)
- Provides error handling for architecture mismatches

## Training Parameters

### Model 1
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 64
- Data augmentation: Yes

### Model 2
- Learning rate: 0.0005
- Weight decay: 0.0002
- Batch size: 128
- Data augmentation: Yes

### Model 3 (Both Versions)
- Learning rate: 0.0005
- Weight decay: 0.0003
- Batch size: 128
- Data augmentation: One with, one without

## Performance

The models typically achieve:
- Model 1: ~95-97% test accuracy
- Model 2: ~97-98% test accuracy  
- Model 3 (with augmentation): ~98-99% test accuracy
- Model 3 (without augmentation): ~97-98% test accuracy

The comparison between Model 3 with and without augmentation demonstrates the effectiveness of data augmentation techniques.

Performance may vary based on hardware and training conditions.

## Data Separation

The system properly separates data to prevent overfitting:
- **Training Data**: 90% of MNIST training set with augmentation
- **Validation Data**: 10% of MNIST training set without augmentation
- **Test Data**: Official MNIST test set (completely separate) without augmentation

## Early Stopping

All models use early stopping with:
- **Patience**: 5 epochs
- **Monitoring**: Validation accuracy
- **Best model saving**: Automatically saves the best model during training

## File Structure

```
mnist_del2/
├── main.py              # Main training script
├── draw_digit.py        # Digit drawing utility
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # MNIST dataset (downloaded automatically)
├── drawn_digits/       # Hand-drawn digits for testing
├── outputs/            # Training outputs and results
└── venv/               # Virtual environment (if used)
```

## Troubleshooting

### No Models Found
If you get "No best model found!" when testing with drawn images:
1. Make sure you've trained models first (select option 1)
2. Check that training completed successfully
3. Verify models are saved in the outputs directory

### Model Loading Issues
The system automatically handles:
- Different model architectures
- Missing or extra parameters
- Batch normalization mismatches

### Memory Issues
If you run out of memory:
1. Reduce batch size in the code
2. Use CPU instead of GPU
3. Close other applications

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes. 