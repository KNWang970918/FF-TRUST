import torch
import torch.nn as nn
from models.deepsleepnet import DeepSleepEpochNet, DeepSleepNet


class DeepSleepEncoder(nn.Module):
    def __init__(self, params):
        super(DeepSleepEncoder, self).__init__()
        self.params = params

        self.seq_len = getattr(params, "seq_len", 20)
        self.in_channels = getattr(params, "in_channels", 2)
        self.input_length = getattr(params, "input_length", 3000)
        self.hidden_dim = getattr(params, "latent_dim", 512)
        self.dropout = getattr(params, "dropout", 0.0)
        self.num_rnn_layers = getattr(params, "lstm_layers", 1)

        self.epoch_encoder = DeepSleepEpochNet(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.seq_encoder = DeepSleepNet(
            hidden_dim=self.hidden_dim,
            num_rnn_layers=self.num_rnn_layers,
            dropout=self.dropout,
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [B, L, C, T], got {tuple(x.shape)}")

        bz, seq_len, channels, signal_len = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got channels={channels}"
            )

        x = x.reshape(bz * seq_len, channels, signal_len)

        x = self.epoch_encoder(x)                  

        x_epoch = x.reshape(bz, seq_len, -1)

        x_seq = self.seq_encoder(x_epoch)          

        mu = self.fc_mu(x_seq)                     
        return mu