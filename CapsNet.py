import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsule_dim = capsule_dim
        self.capsules = nn.ModuleDict({
            f'capsule_{i}': nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
            for i in range(capsule_dim)
        })

    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules.values()]
        outputs = torch.cat(outputs, dim=-1)
        outputs = squash(outputs)
        return outputs

class DigitCapsules(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_iterations):
        super(DigitCapsules, self).__init__()
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(torch.randn(out_capsules, in_capsules, in_dim, out_dim))

    # routing algorithm
    def forward(self, x):
        priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
        logits = torch.zeros(*priors.size(), device=x.device)
        for i in range(self.num_iterations):
            probs = torch.softmax(logits, dim=2)  # probs = c, logits = b (from the paper)
            outputs = squash((probs * priors).sum(dim=2, keepdim=True))
            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits
        return outputs

class Decoder(nn.Module):
    def __init__(self, num_classes, img_channels, img_width):
        super(Decoder, self).__init__()
        self.img_channels = img_channels
        self.img_width = img_width
        self.fc1 = nn.Linear(16 * num_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, img_channels * img_width**2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

class CapsNet(nn.Module):
    def __init__(self, num_classes=10, img_channels=1, img_width=28):
        super(CapsNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(in_channels=256, out_channels=32, capsule_dim=8, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(in_capsules=(32*6*6), in_dim=8, out_capsules=10, out_dim=16, num_iterations=3)
        self.decoder = Decoder(num_classes=num_classes, img_channels=img_channels, img_width=img_width)

    def forward(self, x, y=None, all_reconstructions=False, perturb=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        vecs = self.digit_capsules(x)
        x = vecs
        x = x.view(self.num_classes, -1, 16).transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=1)
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = F.one_hot(max_length_indices, num_classes=self.num_classes).float().to(classes.device)
        else:
            y = F.one_hot(y, num_classes=self.num_classes).float().to(classes.device)
        reconstruction = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))
        ret = [classes, reconstruction]
        if all_reconstructions:
            reconstructions = []
            for i in range(self.num_classes):
                mask = F.one_hot(torch.tensor(i), num_classes=self.num_classes).float().to(x.device)
                reconstructions.append(self.decoder((x * mask[:, :, None]).view(x.size(0), -1)))
            reconstructions = torch.cat(reconstructions,  dim=0)
            ret.append(reconstructions)
        if perturb is not None:
            r = torch.arange(-5, 6, 1)/20
            index = max_length_indices.data[perturb]
            x = x[perturb:perturb+1]
            y = y[perturb:perturb+1]
            vec = (x * y[:, :, None]).view(x.size(0), -1)
            vec = vec.repeat(len(r) * 16, 1)
            for feature_index in range(16):
                for i, val in enumerate(r):
                    vec[len(r)*feature_index+i, 16*index+feature_index] = val
            perturbations = self.decoder(vec)
            ret.append(perturbations)
        return tuple(ret)

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, classes, reconstructions):
        labels = F.one_hot(labels, num_classes=classes.size(1)).float().to(classes.device)
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum(dim=1).mean()
        images = images.view(images.size(0), -1)
        reconstructions = reconstructions.view(reconstructions.size(0), -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

