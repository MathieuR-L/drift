
import math
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def save_image_grid(images, path, nrow=8):
    """
    Save a grid of images to a file.
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        path (str): Path to save the image.
        nrow (int): Number of images per row.
    """
    # Denormalize
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    vutils.save_image(images, path, nrow=nrow)

def plot_loss(losses, path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Drift Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
