import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from CapsNet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # configure cuda gpu

batch_size = 32 # number of images processed in a single pass
img_channels = 1 # 1 channel for monochrome
num_classes = 10
num_iterations = 3
img_width = 28
kernel_size = 9

# Load MNIST dataset
transform = transforms.Compose([ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # convert images to PyTorch tensors and normalizes values.
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True) # download training dataset
test_dataset = MNIST(root='./data', train=False, transform=transform, download=False) # define testing dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # create a shuffled dataloader for training
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # create a dataloader for testing

# Create the CapsuleNet instance
capsule_net = CapsuleNet(img_channels, num_classes, num_iterations, img_width, kernel_size).to(device)
capsule_loss = CapsuleLoss()

# Training settings
num_epochs = 10
learning_rate = 0.001

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(capsule_net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

# Training loop
def train(model, loss_fn, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output, reconstructions = model(data)

        # Compute the loss
        loss = loss_fn(data, target, output, reconstructions)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    average_loss = total_loss / len(train_loader.dataset)
    print(f"Average Training Loss: {average_loss:.4f}")


# Testing loop
def test(model, loss_fn, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, reconstructions = model(data)
            loss = loss_fn(data, target, output, reconstructions)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    average_loss = total_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Average Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")


# Training and Testing
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(capsule_net, capsule_loss, train_loader, optimizer)
    test(capsule_net, capsule_loss, test_loader)
    scheduler.step()

# Save the trained model
torch.save(capsule_net.state_dict(), 'capsule_net_model.pth')