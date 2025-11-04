import torch
import torchvision
from dataloader import RunwayDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("SAVING CHECK POINT......")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("......LOADING CHECKPOINT")
    model.load_state_dict(checkpoint["state_dict"])