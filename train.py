import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import os
from utils import (
    load_checkpoint, 
    save_checkpoint,
    save_predictions_as_imgs,
    check_accuracy,
    get_loaders
)
from utils import DiceLoss


#HYPER-PARAMETERS
LEARNING_RATE = 3e-4
device = ('cuda' if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = r"D:\RITHAM\CODES\Python\deep_learning\projects\runaway_detector\data\train_images"
TRAIN_MASK_DIR = r"D:\RITHAM\CODES\Python\deep_learning\projects\runaway_detector\data\train_masks"
TEST_IMG_DIR = r"D:\RITHAM\CODES\Python\deep_learning\projects\runaway_detector\data\test_images"
TEST_MASK_DIR = r"D:\RITHAM\CODES\Python\deep_learning\projects\runaway_detector\data\test_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(dim = 1).to(device=device)

        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
            
        ],
    ) 
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
            
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(device=device)
    loss_fn = lambda pred, target: nn.BCEWithLogitsLoss()(pred, target) + DiceLoss()(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn=loss_fn, scaler=scaler)
        
        #Save model

        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(test_loader, model, device=device)

        save_predictions_as_imgs(
            loader=test_loader, model=model, folder="saved_images/", device=device
        )
        

if __name__ == "__main__":
    main()