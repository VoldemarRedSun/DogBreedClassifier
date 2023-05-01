from model_utils import Base, DogBreedDataset, EnsembleModel_mean
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from  pathlib import Path









def fit_model(epochs,  model,  train_loader, val_loader, weight_decay=0, device='cpu',
              opt_func=torch.optim.Adam):
    history = []
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = opt_func(params_to_update, weight_decay=weight_decay)

    for epoch in range(epochs):
        # Training phase
        print(f"epoch {epoch} started")
        model.train()
        train_losses = []
        for img, targets in train_loader:
            img = img.to(device)
            targets = targets.to(device)
            loss = model.train_step(img, targets)
            train_losses.append(loss)

            # calculates gradients
            loss.backward()

            # perform gradient descent and modifies the weights
            optimizer.step()

            # reset the gradients
            optimizer.zero_grad()
        # Validation phase
        with torch.no_grad():
            result = evaluate(model, val_loader, device)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_results(result, epoch=epoch)
            history.append(result)
    return history


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = []
    for img, targets in val_loader:
        img = img.to(device)
        targets = targets.to(device)
        outputs.append(model.val_step(img, targets))
    return model.val_epoch_results(outputs)



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH_MODEL_A = Path.cwd().parent.parent / 'saved_models' / 'modelA.pt'
    PATH_MODEL_B = Path.cwd().parent.parent / 'saved_models' / 'modelB.pt'
    PATH_MODEL_C = Path.cwd().parent.parent / 'saved_models' / 'modelC.pt'
    PATH_ENSEMBLE = Path.cwd().parent.parent / 'saved_models' /'ensemble.pt'
    DATASET_PATH = Path.cwd().parent.parent/'dataset'/'stanford_dogs_dataset'

    dataset = ImageFolder(DATASET_PATH)
    val_pct = 0.3
    val_size = int(len(dataset) * val_pct)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Resize((256, 256)),
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor()

    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    batch_size = 50
    train_dataset = DogBreedDataset(train_ds, train_transform)
    val_dataset = DogBreedDataset(val_ds, val_transform)

    btstrp_dlA = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset, replacement=True))
    btstrp_dlB = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset, replacement=True))
    btstrp_dlC = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset, replacement=True))


    val_dl = DataLoader(val_dataset, batch_size, shuffle = False)
    model_A = torch.load(PATH_MODEL_A).to(device)
    model_B = torch.load(PATH_MODEL_B).to(device)
    model_C = torch.load(PATH_MODEL_C).to(device)
    num_epochs = 10
    weight_decay = 0.01


    fit_model(num_epochs,  model_A,  btstrp_dlA, val_dl, weight_decay, device)
    model_A.save_model(PATH_MODEL_A)
    fit_model(num_epochs,  model_A,  btstrp_dlB, val_dl, weight_decay, device)
    model_B.save_model(PATH_MODEL_B)
    fit_model(num_epochs,  model_C,  btstrp_dlC, val_dl, weight_decay, device)
    model_C.save_model(PATH_MODEL_A)
    ensemble = EnsembleModel_mean(model_A, model_B, model_C)
    ensemble.save_model(PATH_ENSEMBLE)

