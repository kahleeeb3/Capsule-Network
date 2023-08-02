import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from CapsNet import *
  
# Training loop
def train(model, loss_fn, train_loader, optimizer, device):
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


def test(model, loss_fn, test_loader, device):
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

def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 64 # This greatly impacts GPU usage!
    learning_rate = 0.001

    # 1. Load MNIST dataset
    transform = transforms.Compose([ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # convert images to PyTorch tensors and normalizes values.
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True) # download training dataset
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=False) # define testing dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # create a shuffled dataloader for training
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # create a dataloader for testing

    # 2. Initialize the CapsNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CapsNet().to(device)
    capsule_loss = CapsuleLoss()

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    # Training and Testing
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, capsule_loss, train_loader, optimizer, device)
        # mixed_precision_train(model, capsule_loss, train_loader, optimizer, device)
        test(model, capsule_loss, test_loader, device)
        scheduler.step()


if __name__ == "__main__":
    main()
