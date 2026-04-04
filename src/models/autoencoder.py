from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from src.utils.logging import get_logger

log = get_logger(__name__)


class LSTMEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:

        _, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]                     
        latent = self.fc(self.dropout(last_hidden))  
        return latent, (hidden, cell)


class LSTMDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(latent).unsqueeze(1)              
        x = x.repeat(1, self.seq_len, 1)                 
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)                        
        reconstruction = self.fc_out(lstm_out)            
        return reconstruction

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.dropout_p  = dropout

        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

        self._init_weights()
        log.debug(
            "LSTMAutoencoder: input=%d hidden=%d latent=%d seq_len=%d "
            "layers=%d dropout=%.2f params=%d",
            input_dim, hidden_dim, latent_dim, seq_len,
            num_layers, dropout, self.count_parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.permute(0, 2, 1)                          
        latent, _ = self.encoder(x_seq)                      
        reconstruction = self.decoder(latent)                 
  
        return reconstruction.permute(0, 2, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.permute(0, 2, 1)
        latent, _ = self.encoder(x_seq)
        return latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            recon = self.forward(x)
            
            errors = ((x - recon) ** 2).mean(dim=[1, 2])  
        return errors

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_error(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config: SimpleNamespace, seq_len: int) -> "LSTMAutoencoder":
        return cls(
            input_dim=getattr(config.autoencoder, "input_dim", 1),
            hidden_dim=getattr(config.autoencoder, "hidden_dim", 64),
            latent_dim=getattr(config.autoencoder, "latent_dim", 16),
            seq_len=seq_len,
            num_layers=getattr(config.autoencoder, "num_layers", 2),
            dropout=getattr(config.autoencoder, "dropout", 0.2),
        )

    def __repr__(self) -> str:
        return (
            f"LSTMAutoencoder("
            f"input={self.input_dim}, hidden={self.hidden_dim}, "
            f"latent={self.latent_dim}, seq_len={self.seq_len}, "
            f"params={self.count_parameters():,})"
        )