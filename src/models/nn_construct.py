from model_utils import Base, DogBreedResnet50
from  pathlib import Path




def set_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.network.fc.parameters():
        param.requires_grad = True
    return model


if __name__ == "__main__":
    PATH_MODEL_A = Path.cwd().parent.parent/'saved_models'/'modelA.pt'
    PATH_MODEL_B  = Path.cwd().parent.parent/'saved_models'/'modelB.pt'
    PATH_MODEL_C = Path.cwd().parent.parent/'saved_models'/'modelC.pt'
    res50_modelA = set_grad(DogBreedResnet50())
    res50_modelB = set_grad(DogBreedResnet50())
    res50_modelC = set_grad(DogBreedResnet50())
    res50_modelA.save_model(PATH_MODEL_A)
    res50_modelB.save_model(PATH_MODEL_B)
    res50_modelC.save_model(PATH_MODEL_C)
