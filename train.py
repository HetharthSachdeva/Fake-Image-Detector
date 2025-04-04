import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import FakeImageDetector
from data_loader import get_data_loaders
from utils import save_model, plot_training_metrics
from adversarial import fgsm_attack, pgd_attack

def train_model(data_dir, num_epochs=30, batch_size=64, learning_rate=0.001, 
                save_dir='weights', adversarial_training=True, epsilon=0.03):
    """
    Train the fake image detection model with optional adversarial training.
    
    Args:
        data_dir: Directory containing the dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save model weights
        adversarial_training: Whether to use adversarial examples during training
        epsilon: Perturbation strength for adversarial examples
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    
    # Initialize model
    model = FakeImageDetector().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with clean images
            outputs_clean = model(images)
            loss_clean = criterion(outputs_clean, labels)
            
            # If using adversarial training, add adversarial examples
            if adversarial_training and torch.rand(1).item() > 0.5:  # 50% chance
                # Generate adversarial examples
                with torch.enable_grad():
                    adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
                
                # Forward pass with adversarial images
                outputs_adv = model(adv_images)
                loss_adv = criterion(outputs_adv, labels)
                
                # Combined loss (weighted equally)
                loss = 0.5 * loss_clean + 0.5 * loss_adv
            else:
                # Just use clean loss
                loss = loss_clean
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics (using clean outputs for metrics calculation)
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs_clean, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_targets, train_preds)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average='binary')
        val_recall = recall_score(val_targets, val_preds, average='binary')
        val_f1 = f1_score(val_targets, val_preds, average='binary')
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Adjust learning rate
        scheduler.step(epoch_val_loss)
        
        # Save model if it's the best so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_model(model, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved new best model with val loss: {best_val_loss:.4f}")
        
        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(model, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Save final model
    save_model(model, os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)
    
    return model