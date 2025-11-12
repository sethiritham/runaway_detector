import torch
import torchvision
from dataloader import RunwayDataset
import torch.nn as nn
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("SAVING CHECK POINT......")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("......LOADING CHECKPOINT")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RunwayDataset(
        img_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = RunwayDataset(
        img_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader

def check_accuracy(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with torch.no_grad():
        for x, y in loader:
            x  = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            intersection = (preds * y).sum()
            dice_score += (2* intersection + 1e-8) /  (preds.sum() + y.sum() + 1e-8)
    print(f"GOT {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100}")
    print(f"DICE SCORE: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds*targets).sum()
        return 1 - (2*intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
