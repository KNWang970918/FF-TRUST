import torch
import torch.nn as nn
from models.encoder import Encoder
from models.deepsleep_encoder import DeepSleepEncoder
from models.decoder import Decoder


class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()

        encoder_type = getattr(params, "encoder_type", "transformer")

        if encoder_type == "transformer":
            self.encoder = Encoder(params)
        elif encoder_type == "deepsleep":
            self.encoder = DeepSleepEncoder(params)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.decoder = Decoder(params)

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.encoder(x)
        recon = self.decoder(mu)
        return recon, mu