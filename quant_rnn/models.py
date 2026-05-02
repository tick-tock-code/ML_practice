from __future__ import annotations

import torch
from torch import nn


class RNNRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        blocks = []
        for layer in range(num_layers):
            blocks.append(
                TemporalBlock(
                    channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, time, features] -> [batch, features, time]
        y = x.transpose(1, 2)
        y = self.input_projection(y)
        y = self.tcn(y)
        return self.head(y[:, :, -1]).squeeze(-1)


MODEL_REGISTRY = {
    "rnn": RNNRegressor,
    "lstm": LSTMRegressor,
    "tcn": TCNRegressor,
}


def create_model(
    name: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    tcn_kernel_size: int = 3,
) -> nn.Module:
    model_name = name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    if model_name == "tcn":
        return TCNRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
    return MODEL_REGISTRY[model_name](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
