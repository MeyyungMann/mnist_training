#!/usr/bin/env python3
"""
MNIST Handwritten Digit Classification
Assignment-1-part-2: Power-ups

This script trains and evaluates CNN models on the MNIST dataset.
"""

import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import datetime
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Global variables for directory paths (will be set in main function)
RUN_TIMESTAMP = None
BASE_SAVE_DIR = None
MODEL_SAVE_DIR = None
PLOT_SAVE_DIR = None
LOG_DIR = None
DIRECTORIES_CREATED = False
SCRIPT_STARTED = False
device = None

def load_data(
    batch_size=64,
    use_augmentation=False,
    rotation_degrees=10,
    translation=(0.1, 0.1),
    scale=(0.9, 1.1),
    random_erasing_prob=0.0,
    random_crop_padding=None,
    num_workers=4,
    pin_memory=True,
    val_split=0.1
):
    """
    Load and prepare MNIST dataset with configurable augmentation.
    Always use augmentation only for training set, never for val/test.
    Returns: train_loader, val_loader, test_loader
    """
    # Basic transform for val/test
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Augmentation for train
    if use_augmentation:
        augmentation_layers = []
        if rotation_degrees > 0:
            augmentation_layers.append(transforms.RandomRotation(rotation_degrees))
        if translation != (0, 0) or scale != (1, 1):
            augmentation_layers.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=translation,
                    scale=scale
                )
            )
        if random_crop_padding is not None:
            augmentation_layers.append(
                transforms.RandomCrop(
                    28,
                    padding=random_crop_padding,
                    padding_mode='edge'
                )
            )
        augmentation_layers.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if random_erasing_prob > 0:
            augmentation_layers.append(
                transforms.RandomErasing(p=random_erasing_prob)
            )
        train_transform = transforms.Compose(augmentation_layers)
    else:
        train_transform = basic_transform

    # Load full train set with train_transform
    full_train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=train_transform
    )
    # Split into train/val
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    # For val set, override transform to basic (no augmentation)
    val_dataset.dataset.transform = basic_transform
    # Test set (always basic transform)
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=basic_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader

class OneLayerCNN(nn.Module):
    def __init__(self, dropout_rate=0.0, use_batch_norm=False, use_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_pooling = use_pooling
        self.num_conv_layers = 1
        self.filters_per_layer = [32]
        if use_pooling:
            self.fc1 = nn.Linear(32 * 14 * 14, 128)
        else:
            self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.use_pooling:
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 32 * 14 * 14)
        else:
            x = x.view(-1, 32 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TwoLayerCNN(nn.Module):
    def __init__(self, dropout_rate=0.0, use_batch_norm=False, use_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else None
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_pooling = use_pooling
        self.num_conv_layers = 2
        self.filters_per_layer = [32, 64]
        if use_pooling:
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
        else:
            self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.use_pooling:
            x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if self.use_pooling:
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64 * 7 * 7)
        else:
            x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ThreeLayerCNN(nn.Module):
    def __init__(self, dropout_rate=0.0, use_batch_norm=False, use_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else None
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else None
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_pooling = use_pooling
        self.num_conv_layers = 3
        self.filters_per_layer = [32, 64, 128]
        
        if use_pooling:
            self.fc1 = nn.Linear(128 * 3 * 3, 128)
        else:
            self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.use_pooling:
            x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if self.use_pooling:
            x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        if self.use_pooling:
            x = F.max_pool2d(x, 2)
        
        if self.use_pooling:
            x = x.view(-1, 128 * 3 * 3)
        else:
            x = x.view(-1, 128 * 28 * 28)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20, model_name="model", params_dict=None, patience=5, lr_scheduler=None):
    """
    Train the model and return training history
    """
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Learning rate scheduling
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model with descriptive name
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'params': params_dict
            }
            # Create descriptive filename with accuracy
            best_model_filename = f'{model_name}_best_acc_{val_acc:.2f}.pth'
            torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, best_model_filename))
            print(f'New best model saved: {best_model_filename} (Validation Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
        
        # Save epoch checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'params': params_dict
        }
        torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, f'{model_name}_epoch_{epoch+1}.pth'))
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    return model, history

def plot_training_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{model_name}_training_history.png'))
    plt.show()

def plot_confusion_matrix(model, test_loader, model_name="model"):
    """Plot confusion matrix with both counts and percentages"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'{model_name} - Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot percentages
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title(f'{model_name} - Confusion Matrix (Percentages)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, f'{model_name}_confusion_matrix.png'))
    plt.show()
    
    # Print summary statistics
    print(f"\n{model_name} - Confusion Matrix Summary:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Overall accuracy: {100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels):.2f}%")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = 100 * np.sum((np.array(all_preds) == np.array(all_labels)) & class_mask) / np.sum(class_mask)
            print(f"  Digit {i}: {class_acc:.2f}% ({np.sum(class_mask)} samples)")

def train_model3_comparison():
    """Train 3-layer model with and without augmentation for comparison"""
    print("\nTraining 3-layer model (model 3) with augmentation...")
    model_aug = ThreeLayerCNN(dropout_rate=0.3, use_batch_norm=True).to(device)
    train_loader_aug, val_loader_aug, test_loader_aug = load_data(batch_size=128, use_augmentation=True)
    optimizer_aug = optim.Adam(model_aug.parameters(), lr=0.0005, weight_decay=0.0003)
    criterion = nn.CrossEntropyLoss()
    
    params_dict_aug = {
        'num_conv_layers': 3,
        'filters_per_layer': [32, 64, 128],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'use_pooling': True,
        'learning_rate': 0.0005,
        'weight_decay': 0.0003,
        'batch_size': 128,
        'use_augmentation': True
    }
    
    model_aug, history_aug = train_model(model_aug, train_loader_aug, val_loader_aug, optimizer_aug, criterion, 
                                       epochs=20, model_name="model_3layer_aug", params_dict=params_dict_aug)
    plot_training_history(history_aug, "model_3layer_aug")
    plot_confusion_matrix(model_aug, test_loader_aug, model_name="model_3layer_aug")

    print("\nTraining 3-layer model (model 3) without augmentation...")
    model_noaug = ThreeLayerCNN(dropout_rate=0.3, use_batch_norm=True).to(device)
    train_loader_noaug, val_loader_noaug, test_loader_noaug = load_data(batch_size=128, use_augmentation=False)
    optimizer_noaug = optim.Adam(model_noaug.parameters(), lr=0.0005, weight_decay=0.0003)
    
    params_dict_noaug = {
        'num_conv_layers': 3,
        'filters_per_layer': [32, 64, 128],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'use_pooling': True,
        'learning_rate': 0.0005,
        'weight_decay': 0.0003,
        'batch_size': 128,
        'use_augmentation': False
    }
    
    model_noaug, history_noaug = train_model(model_noaug, train_loader_noaug, val_loader_noaug, optimizer_noaug, criterion, 
                                           epochs=20, model_name="model_3layer_noaug", params_dict=params_dict_noaug)
    plot_training_history(history_noaug, "model_3layer_noaug")
    plot_confusion_matrix(model_noaug, test_loader_noaug, model_name="model_3layer_noaug")

def display_sample_data():
    """Display sample data from the dataset"""
    print("\n" + "="*60)
    print("Displaying Sample Data from MNIST Dataset")
    print("="*60)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size=16, use_augmentation=False)
    
    # Get a batch of data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Create figure to display images
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Sample MNIST Digits', fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # Get image and denormalize
        img = images[i, 0].numpy()
        img = img * 0.3081 + 0.1307  # Denormalize
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Digit: {labels[i].item()}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'sample_data.png'))
    plt.show()
    
    print(f"Sample data saved to: {os.path.join(PLOT_SAVE_DIR, 'sample_data.png')}")
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

def create_saliency_map(model, image, target_class=None):
    """Create saliency map for a given image and model"""
    model.eval()
    
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    image = image.to(device)
    image.requires_grad_(True)
    
    # Forward pass
    output = model(image)
    
    # If target class not specified, use predicted class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients
    gradients = image.grad.data
    
    # Create saliency map
    saliency_map = torch.abs(gradients).squeeze()
    
    return saliency_map.cpu().numpy()

def get_best_model():
    """Get the best available trained model automatically"""
    # Search in all timestamped directories
    all_model_files = []
    outputs_dir = 'outputs'
    if os.path.exists(outputs_dir):
        for run_dir in os.listdir(outputs_dir):
            if run_dir.startswith('run_'):
                run_path = os.path.join(outputs_dir, run_dir, 'models')
                if os.path.exists(run_path):
                    model_files = glob.glob(os.path.join(run_path, "*_best_*.pth"))
                    for model_file in model_files:
                        # Extract run info and model info
                        run_info = run_dir.replace('run_', '')
                        model_name = os.path.basename(model_file)
                        all_model_files.append({
                            'path': model_file,
                            'run': run_info,
                            'name': model_name,
                            'full_path': model_file
                        })
    
    if not all_model_files:
        print("No trained models found! Please train models first (option 1).")
        return None
    
    # Sort by accuracy (extract from filename)
    def extract_accuracy(model_info):
        try:
            acc_str = model_info['name'].split('acc_')[-1].split('.pth')[0]
            return float(acc_str)
        except:
            return 0.0
    
    # Sort by accuracy (highest first) and return the best
    all_model_files.sort(key=extract_accuracy, reverse=True)
    best_model = all_model_files[0]
    
    accuracy = extract_accuracy(best_model)
    run_date = best_model['run'][:8]  # YYYYMMDD
    run_time = best_model['run'][9:15]  # HHMMSS
    formatted_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"
    formatted_time = f"{run_time[:2]}:{run_time[2:4]}:{run_time[4:6]}"
    
    print(f"Best model found: {best_model['name']}")
    print(f"Accuracy: {accuracy:.2f}% | Date: {formatted_date} {formatted_time}")
    print(f"Run ID: {best_model['run']}")
    
    return best_model['full_path']

def load_model_from_checkpoint(model_path):
    """Load a model from checkpoint file"""
    print(f"Loading model: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get saved parameters
    saved_params = checkpoint.get('params', {})
    
    # Determine model type from filename and recreate with correct parameters
    if '1layer' in model_path:
        dropout_rate = saved_params.get('dropout_rate', 0.0)
        use_batch_norm = saved_params.get('use_batch_norm', False)
        model = OneLayerCNN(dropout_rate=dropout_rate, use_batch_norm=use_batch_norm).to(device)
    elif '2layer' in model_path:
        dropout_rate = saved_params.get('dropout_rate', 0.0)
        use_batch_norm = saved_params.get('use_batch_norm', False)
        model = TwoLayerCNN(dropout_rate=dropout_rate, use_batch_norm=use_batch_norm).to(device)
    elif '3layer' in model_path:
        dropout_rate = saved_params.get('dropout_rate', 0.0)
        use_batch_norm = saved_params.get('use_batch_norm', False)
        model = ThreeLayerCNN(dropout_rate=dropout_rate, use_batch_norm=use_batch_norm).to(device)
    else:
        print("Could not determine model type from filename")
        return None, None
    
    # Load the state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully!")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("This might be due to architecture mismatch. Trying to load with strict=False...")
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Model loaded with warnings (some parameters may not match)")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None, None
    
    model.eval()
    print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint

def test_with_drawn_images():
    """Test the best model with drawn images and show saliency maps"""
    print("\n" + "="*60)
    print("Testing Best Model with Drawn Images")
    print("="*60)
    
    # Get the best model automatically
    model_path = get_best_model()
    if model_path is None:
        return
    
    # Load the selected model
    model, checkpoint = load_model_from_checkpoint(model_path)
    if model is None:
        return
    
    # Check if drawn digit exists
    drawn_digit_path = "drawn_digits/drawn_digit.png"
    if not os.path.exists(drawn_digit_path):
        print(f"\nNo drawn digit found at {drawn_digit_path}")
        print("Please run 'python draw_digit.py' first to create a digit.")
        return
    
    # Load and preprocess the drawn digit
    img = Image.open(drawn_digit_path).convert('L')  # Convert to grayscale
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    print(f"\nPrediction Results:")
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Show all class probabilities
    print(f"\nAll class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  Digit {i}: {prob.item():.2%}")
    
    # Create saliency map
    saliency_map = create_saliency_map(model, input_tensor, predicted_class)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original drawn digit (detach to avoid gradient issues)
    original_img = input_tensor.detach()[0, 0].cpu().numpy()
    original_img = original_img * 0.3081 + 0.1307  # Denormalize
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Drawn Digit\nPredicted: {predicted_class}')
    axes[0].axis('off')
    
    # Saliency map
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map\n(Important regions highlighted)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_img, cmap='gray', alpha=0.7)
    axes[2].imshow(saliency_map, cmap='hot', alpha=0.3)
    axes[2].set_title('Saliency Overlay\n(Original + Saliency)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'drawn_digit_analysis.png'))
    plt.show()
    
    print(f"\nAnalysis saved to: {os.path.join(PLOT_SAVE_DIR, 'drawn_digit_analysis.png')}")
    
    # Test with multiple drawn digits if available
    drawn_digits = glob.glob("drawn_digits/*.png")
    if len(drawn_digits) > 1:
        print(f"\nFound {len(drawn_digits)} drawn digits. Testing all...")
        
        results = []
        for digit_file in drawn_digits:
            try:
                img = Image.open(digit_file).convert('L')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
                
                results.append({
                    'file': digit_file,
                    'predicted': predicted_class,
                    'confidence': confidence
                })
                
                print(f"  {digit_file}: Predicted {predicted_class} (confidence: {confidence:.2%})")
                
            except Exception as e:
                print(f"  Error processing {digit_file}: {e}")
        
        # Calculate overall confidence
        if results:
            print(f"\nOverall confidence: {np.mean([r['confidence'] for r in results]):.2%}")

def draw_and_test():
    """Draw a digit and test it with the best model"""
    print("\n" + "="*60)
    print("Draw and Test Digit")
    print("="*60)
    
    # Get the best model automatically
    model_path = get_best_model()
    if model_path is None:
        return
    
    # Load the selected model
    model, checkpoint = load_model_from_checkpoint(model_path)
    if model is None:
        return
    
    # Now draw the digit
    print("\nOpening drawing window...")
    print("Draw a digit (0-9) in the popup window, then close it to test.")
    
    # Run the drawing utility
    import subprocess
    import sys
    
    try:
        # Run draw_digit.py as a subprocess
        result = subprocess.run([sys.executable, 'draw_digit.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("Digit drawn successfully!")
        else:
            print("Drawing completed or window closed.")
            
    except subprocess.TimeoutExpired:
        print("Drawing window timed out or was closed.")
    except Exception as e:
        print(f"Error running drawing utility: {e}")
        print("Please run 'python draw_digit.py' manually and then test.")
        return
    
    # Check if drawn digit exists
    drawn_digit_path = "drawn_digits/drawn_digit.png"
    if not os.path.exists(drawn_digit_path):
        print(f"\nNo drawn digit found at {drawn_digit_path}")
        print("Please try drawing again or run 'python draw_digit.py' manually.")
        return
    
    # Load and preprocess the drawn digit
    img = Image.open(drawn_digit_path).convert('L')  # Convert to grayscale
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    print(f"\n" + "="*40)
    print(f"PREDICTION RESULTS")
    print(f"="*40)
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Show all class probabilities
    print(f"\nAll class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  Digit {i}: {prob.item():.2%}")
    
    # Create saliency map
    saliency_map = create_saliency_map(model, input_tensor, predicted_class)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original drawn digit (detach to avoid gradient issues)
    original_img = input_tensor.detach()[0, 0].cpu().numpy()
    original_img = original_img * 0.3081 + 0.1307  # Denormalize
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Drawn Digit\nPredicted: {predicted_class}')
    axes[0].axis('off')
    
    # Saliency map
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('Saliency Map\n(Important regions highlighted)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_img, cmap='gray', alpha=0.7)
    axes[2].imshow(saliency_map, cmap='hot', alpha=0.3)
    axes[2].set_title('Saliency Overlay\n(Original + Saliency)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'drawn_digit_analysis.png'))
    plt.show()
    
    print(f"\nAnalysis saved to: {os.path.join(PLOT_SAVE_DIR, 'drawn_digit_analysis.png')}")
    print(f"Original digit saved to: {drawn_digit_path}")
    
    # Ask if user wants to draw another digit
    while True:
        choice = input("\nDraw another digit? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            draw_and_test()  # Recursive call to draw another digit
            break
        elif choice in ['n', 'no']:
            print("Returning to main menu...")
            break
        else:
            print("Please enter 'y' or 'n'.")

def main():
    """Main function with menu system"""
    global RUN_TIMESTAMP, BASE_SAVE_DIR, MODEL_SAVE_DIR, PLOT_SAVE_DIR, LOG_DIR, DIRECTORIES_CREATED, SCRIPT_STARTED, device
    
    # Prevent multiple executions
    if SCRIPT_STARTED:
        print("Script already started. Exiting...")
        return
    
    SCRIPT_STARTED = True
    
    # Script header
    print("="*60)
    print("MNIST Handwritten Digit Classification Script")
    print("="*60)
    print(f"Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up directories with timestamp - only create once per script run
    if not DIRECTORIES_CREATED:
        RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        BASE_SAVE_DIR = f'outputs/run_{RUN_TIMESTAMP}'
        MODEL_SAVE_DIR = os.path.join(BASE_SAVE_DIR, 'models')
        PLOT_SAVE_DIR = os.path.join(BASE_SAVE_DIR, 'plots')
        LOG_DIR = os.path.join(BASE_SAVE_DIR, 'logs')

        # Create directories if they don't exist
        print(f"Creating output directories for run: {RUN_TIMESTAMP}")
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        print(f"Saving models to: {MODEL_SAVE_DIR}")
        print(f"Saving plots to: {PLOT_SAVE_DIR}")
        print(f"Saving logs to: {LOG_DIR}")
        print(f"All outputs will be saved in: {BASE_SAVE_DIR}")
        
        DIRECTORIES_CREATED = True
    
    print(f"Device: {device}")
    print("MNIST Handwritten Digit Classification")
    print("="*50)
    
    while True:
        print("\nMenu Options:")
        print("1. Train all models")
        print("2. Display sample data")
        print("3. Test best model with drawn images")
        print("4. Draw and test a digit")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            print("Training all models...")
            print("="*50)
            
            # Train Model 1 (1 conv layer)
            print("\nTraining Model 1 (1 conv layer)...")
            model1 = OneLayerCNN(dropout_rate=0.0, use_batch_norm=False).to(device)
            train_loader1, val_loader1, test_loader1 = load_data(batch_size=64, use_augmentation=True)
            optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=0.0001)
            criterion = nn.CrossEntropyLoss()

            params_dict1 = {
                'num_conv_layers': 1,
                'filters_per_layer': [32],
                'dropout_rate': 0.0,
                'use_batch_norm': False,
                'use_pooling': True,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'batch_size': 64,
                'use_augmentation': True
            }

            model1, history1 = train_model(model1, train_loader1, val_loader1, optimizer1, criterion, 
                                        epochs=20, model_name="model_1layer", params_dict=params_dict1)
            plot_training_history(history1, "model_1layer")
            plot_confusion_matrix(model1, test_loader1, model_name="model_1layer")

            # Train Model 2 (2 conv layers)
            print("\nTraining Model 2 (2 conv layers)...")
            model2 = TwoLayerCNN(dropout_rate=0.2, use_batch_norm=True).to(device)
            train_loader2, val_loader2, test_loader2 = load_data(batch_size=128, use_augmentation=True)
            optimizer2 = optim.Adam(model2.parameters(), lr=0.0005, weight_decay=0.0002)
            criterion = nn.CrossEntropyLoss()

            params_dict2 = {
                'num_conv_layers': 2,
                'filters_per_layer': [32, 64],
                'dropout_rate': 0.2,
                'use_batch_norm': True,
                'use_pooling': True,
                'learning_rate': 0.0005,
                'weight_decay': 0.0002,
                'batch_size': 128,
                'use_augmentation': True
            }

            model2, history2 = train_model(model2, train_loader2, val_loader2, optimizer2, criterion, 
                                        epochs=20, model_name="model_2layer", params_dict=params_dict2)
            plot_training_history(history2, "model_2layer")
            plot_confusion_matrix(model2, test_loader2, model_name="model_2layer")

            # Train Model 3 with and without augmentation comparison
            print("\nTraining Model 3 (3 conv layers) with and without augmentation comparison...")
            train_model3_comparison()

            print("\n" + "="*50)
            print("All training completed!")
            print("="*50)
            print("Models and plots saved in:", MODEL_SAVE_DIR)
            print("Check the outputs directory for results.")
            print(f"\nSummary of this training run:")
            print(f"  - Run ID: {RUN_TIMESTAMP}")
            print(f"  - Base directory: {BASE_SAVE_DIR}")
            print(f"  - Models: {MODEL_SAVE_DIR}")
            print(f"  - Plots: {PLOT_SAVE_DIR}")
            print(f"  - Logs: {LOG_DIR}")
            print("="*50)
            
        elif choice == '2':
            display_sample_data()
            
        elif choice == '3':
            test_with_drawn_images()
            
        elif choice == '4':
            draw_and_test()
            
        elif choice == '5':
            print("\nExiting...")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main() 