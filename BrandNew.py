import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        # self.num_capsules = num_capsules
        # self.capsule_dim = capsule_dim

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0) 
            for _ in range(num_capsules)
        ])

    """
    def forward(self, x):
        primary_capsules = [capsule(x) for capsule in self.capsules]
        primary_capsules = torch.stack(primary_capsules, dim=-1)
        primary_capsules = primary_capsules.view(x.size(0), self.num_capsules, -1)
        return primary_capsules
    """


class DigitCapsules(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_iterations):
        super(DigitCapsules, self).__init__()
        # self.in_capsules = in_capsules
        # self.in_dim = in_dim
        # self.out_capsules = out_capsules
        # self.out_dim = out_dim
        # self.num_iterations = num_iterations

        # Swap the dimensions of W back to the original order
        self.W = nn.Parameter(torch.randn(out_capsules, in_capsules, in_dim, out_dim))
        
    """
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.in_capsules, 1, self.in_dim)
        x = x.repeat(1, 1, self.out_capsules, 1)

        W = self.W.view(1, self.out_capsules, self.in_capsules, self.in_dim, self.out_dim)
        W = W.repeat(batch_size, 1, 1, 1, 1)

        # Swap the dimensions of x and W in the matrix multiplication
        u_hat = torch.matmul(x, W)
        u_hat = u_hat.squeeze()

        b = torch.zeros(batch_size, self.out_capsules, self.in_capsules).to(x.device)

        for i in range(self.num_iterations):
            c = torch.softmax(b, dim=1)
            v = torch.sum(c[:, :, :, None] * u_hat[:, None, :, :], dim=2)

            if i < self.num_iterations - 1:
                a = torch.matmul(u_hat[:, None, :, :], v[:, :, :, None]).squeeze()
                b = b + a

        return v
    """


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(in_channels=256, out_channels=8, num_capsules=32, capsule_dim=8, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(in_capsules=32 * 6 * 6, in_dim=8, out_capsules=10, out_dim=16, num_iterations=3)
    
    """
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x
    """
    

def train_capsnet(model, train_loader, num_epochs, criterion, optimizer, device):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Training finished!")

def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # 1. Load MNIST dataset
    transform = transforms.Compose([ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # convert images to PyTorch tensors and normalizes values.
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True) # download training dataset
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=False) # define testing dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # create a shuffled dataloader for training
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # create a dataloader for testing

    # 2. Initialize the CapsNet model
    model = CapsNet()

    # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_capsnet(model, train_loader, num_epochs, criterion, optimizer, device)
    train(model)


if __name__ == "__main__":
    main()
