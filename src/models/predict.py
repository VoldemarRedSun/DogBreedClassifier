
from model_utils import DataToPredict
from torchvision.datasets import ImageFolder
import torch
from  pathlib import Path



def label2breed(labels, dataset_path):
    dataset = ImageFolder(dataset_path)
    breeds = [' '.join(' '.join(name.split('-')[1:]).split('_')) for name in dataset.classes]
    preds = [breeds[label] for label in labels]
    return preds

def save_pred(pred_path, data, breeds):
    with open(pred_path, "w") as file:
        for i in range(len(breeds)):
            file.write(f"on {data.data.imgs[i][0]} is {breeds[i]}\n")

if __name__ =="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_PATH = Path.cwd().parent.parent/'dataset'/'stanford_dogs_dataset'
    DATA_FOR_PREDICT_PATH = Path.cwd().parent.parent/'predictions'/'Data_for_predict'
    MODEL_PATH =  Path.cwd().parent.parent / 'saved_models' /'ensemble.pt'
    PRED_PATH = Path.cwd().parent.parent / 'predictions' /'predicted_breeds'/'predictions.txt'
    data = DataToPredict(DATA_FOR_PREDICT_PATH, size_transform=224, batch_size = 50, device = device)
    model = torch.load(MODEL_PATH).to(device)
    labels = model.predict_label(data)
    preds = label2breed(labels,DATASET_PATH)
    save_pred(PRED_PATH, data, preds)
