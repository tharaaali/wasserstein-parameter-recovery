from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset


class CaloDataset(Dataset):
    """Torch dataset wrapper for event-level calorimeter tensors."""

    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def weight_init(module: nn.Module) -> None:
    """Kaiming/Xavier initialisation compatible with the original code."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        init_fn = nn.init.normal_
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        init_fn = nn.init.xavier_normal_
    elif isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        init_fn = nn.init.xavier_normal_
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init_fn = lambda weight: nn.init.normal_(weight, mean=1, std=0.02)
    elif isinstance(module, nn.Linear):
        init_fn = nn.init.xavier_normal_
    else:
        return

    init_fn(module.weight.data)
    if module.bias is not None:
        nn.init.normal_(module.bias.data)


class Generator(nn.Module):
    """Element-wise learnable aging-factor tensor."""

    def __init__(self, shape: Tuple[int, int, int], eps: float = 1e-8):
        super().__init__()
        self.W = nn.Parameter(torch.ones(shape))
        self.eps = eps

    def forward(self, inputs: torch.Tensor, aged: bool = True) -> torch.Tensor:
        batch_size = inputs.size(0)
        aging_factors = self.W.repeat(batch_size, 1, 1).view(batch_size, *self.W.shape)
        if aged:
            return inputs / aging_factors
        return inputs * aging_factors


class Discriminator(nn.Module):
    """Simple 2-D CNN discriminator operating per calorimeter layer."""

    def __init__(self, in_dim: int = 40, dim: int = 64):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.InstanceNorm2d(cout, affine=True),
                nn.LeakyReLU(0.2),
            )

        self.main = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            block(dim, dim * 2),
            block(dim * 2, dim * 2),
            block(dim * 2, dim),
            nn.Conv2d(dim, 1, 4),
            nn.MaxPool2d(21),
        )
        self.apply(weight_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.main(inputs).view(-1)


def save_checkpoint(
    epoch: int,
    step_cnt: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    out_dir: str | Path,
) -> None:
    state = {
        "epoch": epoch,
        "step_cnt": step_cnt,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": generator_optimizer.state_dict(),
        "d_optimizer_state_dict": discriminator_optimizer.state_dict(),
    }
    torch.save(state, Path(out_dir) / f"checkpoint_epoch_{epoch}.pth")
