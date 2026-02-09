
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=64, root='./data', train=True):
    """
    Creates a DataLoader for the MNIST dataset.
    
    Args:
        batch_size (int): Batch size.
        root (str): Root directory for dataset.
        train (bool): Whether to load training or test set.
        
    Returns:
        DataLoader: The MNIST dataloader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True if train else False,
        num_workers=2,
        pin_memory=True
    )
    
    return loader
