import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): Whether this is training data or testing data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Define paths based on whether it's training or testing
        if is_train:
            self.real_dir = os.path.join(root_dir, 'train', 'REAL')
            self.fake_dir = os.path.join(root_dir, 'train', 'FAKE')
        else:
            self.real_dir = os.path.join(root_dir, 'test', 'REAL')
            self.fake_dir = os.path.join(root_dir, 'test', 'FAKE')
        
        # Get filenames
        self.real_files = [os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir) 
                          if os.path.isfile(os.path.join(self.real_dir, f))]
        self.fake_files = [os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir) 
                          if os.path.isfile(os.path.join(self.fake_dir, f))]
        
        # Combine and create labels
        self.image_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)  # 0 for real, 1 for fake
        
        # Shuffle the dataset
        indices = np.arange(len(self.image_files))
        np.random.shuffle(indices)
        self.image_files = [self.image_files[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Create data loaders for training and testing.
    """
    # Define transforms - note images will be resized to 32x32 as mentioned in the problem
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(data_dir, transform=train_transform, is_train=True)
    test_dataset = ImageDataset(data_dir, transform=test_transform, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader