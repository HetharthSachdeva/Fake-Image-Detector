# k:\Coding\ML\MergeConflict_PVH_ML\adversarial.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon=0.1):
    """
    Fast Gradient Sign Method for generating adversarial examples
    
    Args:
        model: The model to attack
        images: Original images
        labels: True labels
        epsilon: Attack strength parameter
        
    Returns:
        Adversarial examples
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Clone the images and require gradient
    perturbed_images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(perturbed_images)
    
    # Calculate loss
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Create perturbation
    perturbation = epsilon * perturbed_images.grad.sign()
    
    # Add perturbation to original images
    perturbed_images = perturbed_images + perturbation
    
    # Clamp the values to ensure they are in valid image range [0,1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images.detach()

class FeatureNoising(nn.Module):
    """
    A layer that adds Gaussian noise to features during training
    to enhance model robustness against adversarial attacks.
    """
    def __init__(self, std=0.1):
        super(FeatureNoising, self).__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        else:
            return x

def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=10):
    """
    Projected Gradient Descent attack for generating stronger adversarial examples
    
    Args:
        model: The model to attack
        images: Original images
        labels: True labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_iter: Number of iterations
        
    Returns:
        Adversarial examples
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Clone the images
    perturbed_images = images.clone().detach()
    
    for i in range(num_iter):
        # Require gradient
        perturbed_images.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_images)
        
        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        
        # Zero all existing gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Create perturbation
        with torch.no_grad():
            # Get gradient sign
            perturbation = alpha * perturbed_images.grad.sign()
            
            # Add perturbation
            perturbed_images = perturbed_images + perturbation
            
            # Project back to epsilon ball
            delta = torch.clamp(perturbed_images - images, -epsilon, epsilon)
            perturbed_images = torch.clamp(images + delta, 0, 1)
    
    return perturbed_images.detach()