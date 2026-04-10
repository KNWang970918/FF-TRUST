import torch
import torch.nn as nn


class DeepSleepEpochNet(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.short_branch = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=25, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(1),
        )

        self.long_branch = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=400, stride=50, padding=200, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        short_feat = self.short_branch(x).squeeze(-1)   
        long_feat = self.long_branch(x).squeeze(-1)     
        feat = torch.cat([short_feat, long_feat], dim=-1)  
        feat = self.proj(feat) 
        return feat


class DeepSleepNet(nn.Module):


    def __init__(
        self,
        hidden_dim: int = 512,
        num_rnn_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.seq_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,   
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

        self.residual_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_epoch: torch.Tensor):
        torch._assert(
            x_epoch.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim), got {x_epoch.shape}"
        )

        x_seq, _ = self.seq_encoder(x_epoch)       
        x_res = self.residual_proj(x_epoch)        
        out = self.out_proj(x_seq + x_res)        
        return out