
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, content_channels=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # Project and reshape
        self.fc = nn.Linear(z_dim, hidden_dim * 4 * 4 * 4) # -> (B, 256, 4, 4)

        self.net = nn.Sequential(
            # Input: (B, 256, 4, 4)
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False), # -> (B, 128, 8, 8)
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            
            # Input: (B, 128, 8, 8)
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False), # -> (B, 64, 16, 16)
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            
            # Input: (B, 64, 16, 16)
            nn.ConvTranspose2d(hidden_dim, content_channels, 4, 2, 3, bias=False), # -> (B, 1, 28, 28) with padding adjustments
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.hidden_dim * 4, 4, 4)
        return self.net(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
