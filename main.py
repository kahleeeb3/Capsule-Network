import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
def softmax(x, dim=1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return e_x / e_x.sum(dim=dim, keepdim=True)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, num_iterations,\
                 in_channels, out_channels, kernel_size=None, stride=None):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        if num_route_nodes != -1:   # real capsule layer
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, \
                                                          in_channels, out_channels))
        else:  # just convolution
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, \
                           stride=stride, padding=0) for _ in range(num_capsules)])

    def squash(self, tensor, dim=-1):  # take a vector and scale it to have length in [0,1)
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x, save_vecs = False):   # x size = batches, maps, side, side
        if save_vecs:
            vecs = []
        if self.num_route_nodes != -1:   # real capsule layer
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)  # probs = c, logits = b (froom paper)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if save_vecs:
                    vecs.append(outputs)
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:   # takes input from convolution
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        if save_vecs:
            return vecs
        else:
            return outputs

class CapsuleNet(nn.Module):
    def __init__(self, img_channels, num_classes=10, num_iterations=3, img_width=28, kernel_size=9):
        super(CapsuleNet, self).__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=256, \
                               kernel_size=kernel_size, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, \
                                             num_iterations=num_iterations, in_channels=256, \
                                             out_channels=32, kernel_size=kernel_size, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=num_classes, num_route_nodes=\
                                           32*int(((img_width-2*(kernel_size-1))/2)**2), \
                                           num_iterations=num_iterations, in_channels=8, \
                                           out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_channels * img_width**2),
            nn.Sigmoid()
        )

    def forward(self, x, y=None, all_reconstructions=False, perturb=None, save_vecs=False):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        vecs = self.digit_capsules(x, save_vecs=save_vecs)
        if save_vecs:
            x = vecs[-1]
        else:
            x = vecs
        x = x.view(self.num_classes, batch_size, 16).transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=1)
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, \
                                                index=Variable(max_length_indices.data))
            y_was_none = True
        else:
            y_was_none = False

        reconstruction = self.decoder((x * y[:,:, None]).reshape(x.size(0), -1))
        ret = [classes, reconstruction]

        if all_reconstructions:
            reconstructions = []
            for i in range(self.num_classes):
                index = torch.cuda.LongTensor(1)
                index[0] = i
                mask = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=Variable(index))
                reconstructions.append(self.decoder((x * mask[:, :, None]).view(x.size(0), -1)))   
            reconstructions = torch.cat(reconstructions,  dim=0)
            ret.append(reconstructions)

        if y_was_none and perturb is not None:
            r = torch.arange(-5, 6, 1)/20 # -0.25,-0.20,...,0.25
            index = max_length_indices.data[perturb]
            x = x[perturb:perturb+1] # 1 x 10 x 16
            y = y[perturb:perturb+1] # 1 x 10
            vec = (x * y[:, :, None]).view(x.size(0), -1) # 1 x 160
            vec = vec.repeat(len(r) * 16, 1)
            for feature_index in range(16):
                for i, val in enumerate(r):
                    vec[len(r)*feature_index+i, 16*index+feature_index] = val
            perturbations = self.decoder(vec)
            ret.append(perturbations)

        if save_vecs:
            ret.append([vec.view(self.num_classes, batch_size, 16).transpose(0, 1) for vec in vecs])

        return tuple(ret)

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        # Perform one-hot encoding on the target labels
        labels = F.one_hot(labels, num_classes=classes.size(1)).float().to(classes.device)

        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum(dim=1).mean()  # Calculate the mean loss across the batch

        # Flatten images for MSE loss calculation
        images = images.view(images.size(0), -1)

        # Flatten reconstructions for MSE loss calculation
        reconstructions = reconstructions.view(reconstructions.size(0), -1)

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)



# Load MNIST dataset
transform = transforms.Compose([ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform)

batch_size = 32

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create the CapsuleNet instance
img_channels = 1
num_classes = 10
num_iterations = 3
img_width = 28
kernel_size = 9

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
