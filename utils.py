import os
import torch
import matplotlib.pyplot as plt

def save_model(model, path):
    """
    Save model weights to the specified path.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """
    Load model weights from the specified path.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Plot training and validation metrics.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def visualize_predictions(images, true_labels, predicted_labels, num_samples=5, save_path=None):
    """
    Visualize model predictions on sample images.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5  # Denormalize
        
        axes[i].imshow(img)
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        
        title = f"True: {'Fake' if true_labels[i] == 1 else 'Real'}\nPred: {'Fake' if predicted_labels[i] == 1 else 'Real'}"
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()