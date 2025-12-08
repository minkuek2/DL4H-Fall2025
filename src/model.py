import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGClassifier(nn.Module):

    def __init__(self, input_length: int = 187, num_classes: int = 5):
        super().__init__()

        # 1D CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),              # 187 -> 93

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),              # 93 -> 46
        )

        conv_out_len = input_length // 4  # MaxPool1d(2) twice → 187 // 4 ≈ 46
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # 64 * 46
            nn.Linear(64 * conv_out_len, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 187)
        return: logits (batch, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
