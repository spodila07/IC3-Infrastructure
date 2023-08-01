import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import sys

# Open a file for logging
sys.stdout = open("output.log", "w")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If we're using a GPU, print some details about the device.
if device.type == 'cuda':
    # Print the name of the GPU.
    print(torch.cuda.get_device_name(0))
    # Print current GPU memory usage.
    print('Memory Usage:')
    # Print the memory currently allocated on the GPU. 
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # Print the memory cached (but not necessarily allocated) on the GPU.
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Set hyperparameters
latent_dim = 100
hidden_dim = 256
output_dim = 784 # 28*28 because the images are 28x28 pixels
num_epochs = 200
batch_size = 64
lr = 0.0002

# Prepare MNIST dataset
# Use torchvision transforms to convert images to PyTorch tensors and normalize them in range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        # Construct the feedforward neural network
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Pass the input through the network
        return self.main(x)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        # Construct the feedforward neural network
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass the input through the network
        return self.main(x)

# Initialize models
G = Generator(latent_dim, hidden_dim, output_dim).to(device)
D = Discriminator(output_dim, hidden_dim, 1).to(device)

# Use Binary Cross Entropy loss
criterion = nn.BCELoss()
# Use Adam optimizer for both networks
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1).to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

    # Save and plot the generated images every 10 epochs
    if (epoch+1) == 1 or (epoch+1) % 10 == 0:
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        grid = make_grid(fake_images, nrow=10, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig(f'generated_images_epoch_{epoch+1}.png')

sys.stdout.close()

