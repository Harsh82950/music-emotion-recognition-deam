from torch.utils.data import DataLoader, random_split
from src.dataset import DEAMDataset

def get_dataloaders(
    csv_path,
    mel_dir,
    batch_size=4,
    train_split=0.8
):
    dataset = DEAMDataset(csv_path, mel_dir)
    dataset = torch.utils.data.Subset(dataset, range(200))

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.dataset import DEAMDataset
from src.model import EmotionCNN


def get_dataloaders(csv_path, mel_dir, batch_size=16, train_split=0.8):
    dataset = DEAMDataset(csv_path, mel_dir)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for i, (mel, target) in enumerate(loader):
        mel = mel.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for mel, target in loader:
            mel = mel.to(device)
            target = target.to(device)

            output = model(mel)
            loss = criterion(output, target)
            running_loss += loss.item()

    return running_loss / len(loader)


if __name__ == "__main__":
    device = torch.device("cpu")

    train_loader, val_loader = get_dataloaders(
        csv_path="data/labels/deam_song_level_labels.csv",
        mel_dir="data/processed/mel_spectrograms",
        batch_size=16
    )

    model = EmotionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate(
            model, val_loader, criterion, device
        )
        
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
    import os

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/emotion_cnn_regression.pth")

    print("Model saved: models/emotion_cnn_regression.pth")

