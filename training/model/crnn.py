import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_out_channels: int = 512,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # [B, 1, 64, W]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 64, 32, W/2]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 128, 16, W/4]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # [B, 256, 8, W/4]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # [B, 512, 4, W/4]

            nn.AdaptiveAvgPool2d((1, None))
        )

        self.rnn = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Linear(
            2 * rnn_hidden_size,
            num_classes
        )


    def forward(self, x):
        # CNN
        x = self.cnn(x)
        # x: [B, C, 1, W']

        # Remove altura
        x = x.squeeze(2)
        # x: [B, C, W']

        # Transforma em time series
        x = x.permute(0, 2, 1)
        # x: [B, W', C]

        # RNN
        x, _ = self.rnn(x)
        # x: [B, W', 2*hidden]

        # Classifier
        x = self.classifier(x)
        # x: [B, W', num_classes]

        # CTC espera [T, B, C]
        x = x.permute(1, 0, 2)
        # x: [W', B, num_classes]

        return x


