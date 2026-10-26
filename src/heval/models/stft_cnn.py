"""Capacity-matched log-power STFT CNN of the representation analysis (paper Fig. 8a-b, development data only).

Input: the RawCNN-normalised window, STFT per channel (200-sample periodic Hann frames, hop 50, no centring), log power
for 0.5-30 Hz (60 bins x 57 frames), standardised per channel and frequency with training-fold statistics.
Network: four 3x3 Conv2d blocks (channels 2 -> w, 2w, 2.5w, 2w; GELU, GroupNorm(4), 2x2 max-pool between blocks),
per-channel mean and log variance -> Linear 32 + GELU -> Linear 32 + GELU -> Linear 1. Width 32 gives 117,105
parameters (RawCNN: 114,817). Trained with the same early-stopping procedure as RawCNN (ablation_networks.train_net).

Lineage: INTERP v1 (STFTNet, stft_logpow, SpecStats, STFTTensors) of the research archive, tag v0.1.0-paper-review.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_networks import AblationInputs
from .raw_cnn import normalise

N = 3000
FREQ_KEEP = slice(1, 61)                      # 0.5..30.0 Hz at 0.5 Hz resolution
CH_MULT = (1, 2, 2.5, 2)


def stft_logpow(xn):
    """xn: torch (n, 2, 3000) normalised raw -> (n, 2, 60, 57) log power, 0.5..30 Hz."""
    n = xn.shape[0]
    S = torch.stft(xn.reshape(-1, N), 200, 50, window=torch.hann_window(200, device=xn.device), center=False,
                   return_complex=True)
    P = torch.log(S.abs() ** 2 + 1e-6)[:, FREQ_KEEP]
    return P.reshape(n, 2, P.shape[1], P.shape[2])


class STFTNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        c = [2] + [int(round(width * k)) for k in CH_MULT]
        layers = []
        for i in range(len(CH_MULT)):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.GELU(), nn.GroupNorm(4, c[i + 1])]
            if i < len(CH_MULT) - 1:
                layers.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(2 * c[-1], 32)
        self.fuse = nn.Sequential(nn.Linear(32, 32), nn.GELU())
        self.head = nn.Linear(32, 1)

    def forward(self, spec, seq=None, tf=None):
        h = self.body(spec).flatten(2)
        h = F.gelu(self.out(torch.cat([h.mean(-1), torch.log(h.var(-1) + 1e-6)], 1)))
        return self.head(self.fuse(h))


class SpecStats:
    """Training-fold mean / SD of the log-power spectrogram (per channel and frequency)."""

    def __init__(self, raw_uv, rows, scale, device='cpu'):
        self.scale, self.device = scale, device
        P = self._logpow(raw_uv[rows])
        self.mu = P.mean((0, 3), keepdim=True)
        self.sd = P.std((0, 3), keepdim=True) + 1e-6

    @classmethod
    def from_state(cls, scale, mu, sd, device='cpu'):
        s = cls.__new__(cls)
        s.scale, s.device, s.mu, s.sd = scale, device, mu.to(device), sd.to(device)
        return s

    def _logpow(self, raw_uv):
        out = []
        for i in range(0, len(raw_uv), 512):
            x = torch.tensor(normalise(raw_uv[i:i + 512], self.scale), device=self.device)
            out.append(stft_logpow(x))
        return torch.cat(out)

    def spec(self, raw_uv):
        return (self._logpow(raw_uv) - self.mu) / self.sd


class STFTInputs(AblationInputs):
    """AblationInputs whose .raw holds the standardised spectrogram (same rows, labels and RawCNN scale)."""

    def __init__(self, d, tr, device='cpu'):
        super().__init__(d, tr, device=device)
        self.stats = SpecStats(d['raw'], tr, self.scale, device)
        self.raw = self.stats.spec(d['raw'])


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def logits(models, x):
    """Per-model logits (n_models, n) for prepared inputs x (normalised raw or standardised spectrogram), batches of 512."""
    out = []
    for m in models:
        m.eval()
        with torch.no_grad():
            out.append(torch.cat([m(x[i:i + 512])[:, 0] for i in range(0, len(x), 512)]).cpu().numpy().astype(np.float64))
    return np.array(out)
