import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# take a vector and scale it to have length in [0,1)
def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsule_dim = capsule_dim

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0) 
            for _ in range(capsule_dim)
        ])
    
    def forward(self, x):   # x size = batches, maps, side, side
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
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

    # Routing Algorithm
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

class CapsNet(nn.Module):
    def __init__(self, num_classes = 10, img_channels =1, img_width = 28):
        super(CapsNet, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(in_channels=256, out_channels=32, capsule_dim=8, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(in_capsules=(32*6*6), in_dim=8, out_capsules=10, out_dim=16, num_iterations=3)

        self.decoder = nn.Sequential(
                nn.Linear(16 * num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, img_channels * img_width**2),
                nn.Sigmoid()
            )


    def forward(self, x, y=None, all_reconstructions=False, perturb=None):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        vecs = self.digit_capsules(x)
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