import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class DataToPredict(Dataset):
    def __init__(self, img_path, size_transform=100, batch_size = 50, device = 'cuda'):
        super().__init__()
        self.data = ImageFolder(img_path)
        self.transform = transforms.Compose([ transforms.Resize((size_transform, size_transform)),transforms.ToTensor()])
        self.batch_size = batch_size
        self.loader = DataLoader(self, batch_size, shuffle = False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, temp = self.data[idx]
        img = self.transform(img)
        img = img.to(device)
        return img

    def batch_loader(self):
        for img_batch  in self.loader:
            yield img_batch


class DogBreedDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label



class Base(nn.Module):
    # training step
    def train_step(self, img, targets):
        out = self(img)
        loss = F.nll_loss(out, targets)
        return loss

    # validation step
    def val_step(self, img, targets):
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        print(f"local acc = {acc}")
        return {'val_acc': acc.detach(), 'val_loss': loss.detach()}
    # validation epoch end
    def val_epoch_results(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # print result end epoch
    def epoch_results(self, result, epoch = None, mode = 'without train'):
        if mode =='with train':
            print("Epoch [{}] : train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["train_loss"],result["val_loss"],result["val_acc"]))
        if mode == 'without train':
            print("  val_loss: {:.4f}, val_acc: {:.4f}".format( result["val_loss"], result["val_acc"]))

    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            torch.save(self, file)

    @torch.no_grad()
    def predict_label(self, data):
        self.eval()
        preds = torch.Tensor([]).to(device)
        for img in data.batch_loader():
            out = self(img)
            _, batch_preds = torch.max(out, dim=1)
            preds = torch.cat((preds, batch_preds))
        return preds.long()




class DogBreedResnet50(Base):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 120),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)





class DogBreedCustom(Base):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),  # 55

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # 55
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2),  # 26

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),  # 26
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6, stride=2),  # 11

            nn.Flatten(),
            nn.Linear(11 * 11 * 256, 1000),
            nn.ReLU(),
            nn.Linear(1000, 120),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, xb):
        return self.network(xb)


class EnsembleModel_linear(Base):
    def __init__(self, modelA, modelB, modelC):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Sequential( nn.Linear(4 * 3, 4),
        nn.LogSoftmax(dim=1))
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out



class EnsembleModel_mean(Base):
    def __init__(self, modelA, modelB, modelC):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        out = (x1+x2+x3)/3
        return out

