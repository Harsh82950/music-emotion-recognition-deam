import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DEAMDataset
from src.model import EmotionCNN

def evaluate():
    device = torch.device("cpu")

    dataset = DEAMDataset(
        "data/labels/deam_song_level_labels.csv",
        "data/processed/mel_spectrograms"
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("models/emotion_cnn_regression.pth"))
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for mel, target in loader:
            mel, target = mel.to(device), target.to(device)
            output = model(mel)
            loss = criterion(output, target)
            total_loss += loss.item()

    print("Final MSE:", total_loss / len(loader))


if __name__ == "__main__":
    evaluate()
