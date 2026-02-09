
import os
import argparse
import torch
import torch.optim as optim
from dataset import get_mnist_dataloader
from models import Generator, weights_init
from core import drifting_loss
from utils import save_image_grid, plot_loss

def train(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data
    dataloader = get_mnist_dataloader(batch_size=args.batch_size)
    
    # Model
    netG = Generator(z_dim=args.z_dim, hidden_dim=args.hidden_dim).to(device)
    netG.apply(weights_init)
    
    # Optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Training Loop
    global_step = 0
    losses = []
    
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        for i, (data, _) in enumerate(dataloader):
            real_data = data.to(device)
            current_batch_size = real_data.size(0)
            
            # 1. Sample Noise
            z = torch.randn(current_batch_size, args.z_dim, device=device)
            
            # 2. Generate Fake Data
            fake_data = netG(z)
            
            # 3. Compute Loss
            # Note: We can flatten here or inside core. core handles it.
            loss = drifting_loss(fake_data, real_data, temp=args.temp)
            
            # 4. Update Generator
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
            
            # Logging
            losses.append(loss.item())
            global_step += 1
            
            if i % 100 == 0:
                print(f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
            if global_step % args.save_interval == 0:
                with torch.no_grad():
                    # Generate visualization samples
                    viz_z = torch.randn(64, args.z_dim, device=device)
                    viz_data = netG(viz_z)
                    save_image_grid(viz_data, f"{args.output_dir}/step_{global_step}.png")
                    
    # Save Final items
    torch.save(netG.state_dict(), f"{args.output_dir}/netG.pth")
    plot_loss(losses, f"{args.output_dir}/loss.png")
    print("Training Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of generator")
    parser.add_argument("--temp", type=float, default=10.0, help="Temperature for drift kernel")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=500, help="Save interval in steps")
    parser.add_argument("--dry_run", action="store_true", help="Run a single step for testing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        args.epochs = 1
        args.save_interval = 1
        
    train(args)
