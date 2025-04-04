import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import FakeImageDetector
from data_loader import ImageDataset
from adversarial import fgsm_attack, pgd_attack
from utils import load_model

def test_model(model, test_loader, device, adversarial=False, epsilon=0.1):
    """
    Test the model on the provided test loader.
    
    Args:
        model: The model to test
        test_loader: DataLoader for the test set
        device: Device to run the model on
        adversarial: Whether to test with adversarial examples
        epsilon: Perturbation strength for adversarial examples
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            if adversarial:

                images.requires_grad = True
                

                with torch.enable_grad():

                    adv_images = pgd_attack(model, images, labels, epsilon=epsilon)
                

                outputs = model(adv_images)
            else:

                outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, all_preds, all_labels

def generate_submission(model_path, test_data_dir, output_dir='submissions', adversarial=False, epsilon=0.1):
    """
    Generate submission file for a test dataset.
    
    Args:
        model_path: Path to the trained model
        test_data_dir: Directory for test dataset
        output_dir: Directory to save submission files
        adversarial: Whether to test with adversarial examples
        epsilon: Perturbation strength for adversarial examples
    """

    os.makedirs(output_dir, exist_ok=True)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(FakeImageDetector(), model_path, device)

    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    test_dataset = ImageDataset(test_data_dir, transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    metrics, predictions, _ = test_model(model, test_loader, device, adversarial=adversarial, epsilon=epsilon)

    print(f"Test Dataset Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    filenames = [os.path.basename(test_dataset.image_files[j]) for j in range(len(test_dataset))]
    labels = ["Fake" if pred == 1 else "Real" for pred in predictions]
    
    submission_df = pd.DataFrame({
        'Image': filenames,
        'Label': labels
    })
    
    model_name = os.path.basename(model_path).replace('.pth', '')
    dataset_name = os.path.basename(os.path.normpath(test_data_dir))
    test_type = "adversarial" if adversarial else "clean"
    
    output_file = os.path.join(output_dir, f"results_{dataset_name}_{model_name}_{test_type}.csv")
    submission_df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")
    
    return metrics