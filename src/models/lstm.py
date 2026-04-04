from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from src.utils.logging import get_logger

log = get_logger(__name__)


class LSTMAnomalyDetector(nn.Module):

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.num_classes   = num_classes
        self.dropout_p     = dropout
        self.bidirectional = bidirectional

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        fc_input_dim = hidden_dim * directions

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(fc_input_dim)
        self.fc = nn.Linear(fc_input_dim, num_classes)

        self._init_weights()
        log.debug(
            "LSTMAnomalyDetector: input=%d hidden=%d layers=%d classes=%d "
            "dropout=%.2f bidirectional=%s  params=%d",
            input_dim, hidden_dim, num_layers, num_classes,
            dropout, bidirectional, self.count_parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)          
        last_hidden = lstm_out[:, -1, :]    
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)
        logits = self.fc(out)               
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=-1)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x)[:, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @classmethod
    def from_config(cls, config: SimpleNamespace) -> "LSTMAnomalyDetector":
        return cls(
            input_dim=getattr(config.lstm, "input_dim", 1),
            hidden_dim=getattr(config.lstm, "hidden_dim", 64),
            num_layers=getattr(config.lstm, "num_layers", 2),
            num_classes=getattr(config.lstm, "output_dim", 2),
            dropout=getattr(config.lstm, "dropout", 0.2),
            bidirectional=getattr(config.lstm, "bidirectional", False),
        )

    def __repr__(self) -> str:
        return (
            f"LSTMAnomalyDetector("
            f"input={self.input_dim}, hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )