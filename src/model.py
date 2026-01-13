import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- Convolutional Feature Extractor --------
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (128,1292) -> (64,646)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # (64,646) -> (32,323)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)    # (32,323) -> (16,161)
        )

        # -------- Regression Head --------
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 161, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)   # valence, arousal
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    model = EmotionCNN()
    dummy_input = torch.randn(4, 1, 128, 1292)
    out = model(dummy_input)
    print("Output shape:", out.shape)
