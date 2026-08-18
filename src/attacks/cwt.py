"""Component-Wise Transformation for transferable adversarial attacks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


@dataclass(frozen=True)
class CWTConfig:
    """Configuration from the CWT paper, with explicit implementation choices."""

    grid: int = 2
    copies: int = 20
    scale_min: float = 1.0
    scale_max: float = 1.3
    max_angle: float = 26.0
    rotated_blocks: int = 2
    align_corners: bool = False
    antialias: bool = False


class CWT(nn.Module):
    """Create differentiable stochastic CWT views of a ``[B,C,H,W]`` tensor."""

    def __init__(self, config: CWTConfig | None = None) -> None:
        super().__init__()
        self.config = config or CWTConfig()
        if self.config.grid < 1:
            raise ValueError("grid must be positive")
        if self.config.copies < 1:
            raise ValueError("copies must be positive")
        if not 0 <= self.config.rotated_blocks <= self.config.grid**2:
            raise ValueError("rotated_blocks must be between 0 and grid**2")
        if self.config.scale_min < 1.0 or self.config.scale_max < self.config.scale_min:
            raise ValueError("expected 1 <= scale_min <= scale_max")
        if self.config.max_angle < 0:
            raise ValueError("max_angle must be non-negative")

    @property
    def cfg(self) -> CWTConfig:
        """Compatibility alias used by the reference implementation."""

        return self.config

    def _resize(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=self.config.align_corners,
            antialias=self.config.antialias,
        )

    @staticmethod
    def _crop(x: Tensor, output_height: int, output_width: int) -> Tensor:
        height, width = x.shape[-2:]
        if height < output_height or width < output_width:
            raise ValueError("crop is larger than transformed block")
        top = random.randint(0, height - output_height) if height > output_height else 0
        left = random.randint(0, width - output_width) if width > output_width else 0
        return x[..., top : top + output_height, left : left + output_width]

    def _transform_block(self, block: Tensor, rotate: bool) -> Tensor:
        height, width = block.shape[-2:]
        scale = random.uniform(self.config.scale_min, self.config.scale_max)
        small_size = (
            max(1, math.floor(height / scale)),
            max(1, math.floor(width / scale)),
        )
        block = self._resize(block, small_size)
        large_size = (
            max(height, math.floor(height * scale)),
            max(width, math.floor(width * scale)),
        )
        block = self._resize(block, large_size)
        if rotate:
            angle = random.uniform(-self.config.max_angle, self.config.max_angle)
            block = TVF.rotate(
                block,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0,
            )
        return self._crop(block, height, width)

    def forward_once(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("expected [B,C,H,W]")
        _, _, height, width = x.shape
        grid = self.config.grid
        if height % grid or width % grid:
            raise ValueError("H and W must be divisible by grid")

        block_height, block_width = height // grid, width // grid
        rotated = set(torch.randperm(grid * grid)[: self.config.rotated_blocks].tolist())
        blocks: list[Tensor] = []
        index = 0
        for row in range(grid):
            for column in range(grid):
                block = x[
                    ...,
                    row * block_height : (row + 1) * block_height,
                    column * block_width : (column + 1) * block_width,
                ]
                blocks.append(self._transform_block(block, index in rotated))
                index += 1

        rows = [
            torch.cat(blocks[row * grid : (row + 1) * grid], dim=-1)
            for row in range(grid)
        ]
        result = torch.cat(rows, dim=-2)
        if result.shape != x.shape:
            raise RuntimeError(f"shape changed: {tuple(x.shape)} -> {tuple(result.shape)}")
        return result

    def forward(self, x: Tensor, copies: int | None = None) -> Tensor:
        copies = self.config.copies if copies is None else copies
        if copies < 1:
            raise ValueError("copies must be positive")
        return torch.cat([self.forward_once(x) for _ in range(copies)], dim=0)


def _validate_attack_inputs(
    clean: Tensor,
    labels: Tensor,
    epsilon: float,
    steps: int,
) -> None:
    if clean.ndim != 4:
        raise ValueError("clean must have shape [B,C,H,W]")
    if labels.ndim != 1 or labels.shape[0] != clean.shape[0]:
        raise ValueError("labels must have shape [B]")
    if not clean.is_floating_point():
        raise TypeError("clean must be a floating-point tensor")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not torch.isfinite(clean).all():
        raise ValueError("clean contains non-finite values")
    if clean.min().item() < 0.0 or clean.max().item() > 1.0:
        raise ValueError("clean must be in [0,1]")


def _momentum_step(
    clean: Tensor,
    x_adv: Tensor,
    gradient: Tensor,
    momentum: Tensor,
    epsilon: float,
    alpha: float,
    decay: float,
) -> tuple[Tensor, Tensor]:
    denominator = (
        gradient.abs().flatten(1).sum(1).clamp_min(1e-12).view(-1, 1, 1, 1)
    )
    momentum = decay * momentum + gradient / denominator
    with torch.no_grad():
        x_adv = x_adv + alpha * momentum.sign()
        delta = (x_adv - clean).clamp(-epsilon, epsilon)
        x_adv = (clean + delta).clamp(0.0, 1.0)
    return x_adv, momentum


def mifgsm(
    model: nn.Module,
    clean: Tensor,
    labels: Tensor,
    epsilon: float = 16 / 255,
    steps: int = 10,
    mu: float = 1.0,
) -> Tensor:
    """MI-FGSM baseline in pixel space ``[0,1]``."""

    _validate_attack_inputs(clean, labels, epsilon, steps)
    clean = clean.detach().float()
    labels = labels.detach().long()
    x_adv = clean.clone()
    momentum = torch.zeros_like(clean)
    alpha = epsilon / steps

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), labels)
        gradient = torch.autograd.grad(loss, x_adv)[0].float()
        x_adv, momentum = _momentum_step(
            clean, x_adv, gradient, momentum, epsilon, alpha, mu
        )
    return x_adv.detach()


def cwt_gradient(
    model: nn.Module,
    cwt: CWT,
    inputs: Tensor,
    labels: Tensor,
    copy_chunk: int = 4,
) -> Tensor:
    """Return the mean input gradient over exactly ``cwt.config.copies`` views.

    ``inputs`` must require gradients. The helper can be reused inside PGD-like
    attacks that keep their own update and projection rules.
    """

    if not inputs.requires_grad:
        raise ValueError("inputs must require gradients")
    if copy_chunk < 1:
        raise ValueError("copy_chunk must be positive")
    total_copies = cwt.config.copies
    batch_size = inputs.shape[0]
    mean_gradient = torch.zeros_like(inputs, dtype=torch.float32)
    completed = 0
    while completed < total_copies:
        count = min(copy_chunk, total_copies - completed)
        views = cwt(inputs, copies=count)
        targets = labels.repeat(count)
        logits = model(views)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        loss = loss / (total_copies * batch_size)
        gradient = torch.autograd.grad(loss, inputs)[0]
        mean_gradient.add_(gradient.float())
        completed += count
    return mean_gradient


def cwt_mifgsm(
    model: nn.Module,
    cwt: CWT,
    clean: Tensor,
    labels: Tensor,
    epsilon: float = 16 / 255,
    steps: int = 10,
    mu: float = 1.0,
    copy_chunk: int = 4,
) -> Tensor:
    """MI-FGSM with the mean gradient over stochastic CWT copies."""

    _validate_attack_inputs(clean, labels, epsilon, steps)
    if copy_chunk < 1:
        raise ValueError("copy_chunk must be positive")
    clean = clean.detach().float()
    labels = labels.detach().long()
    x_adv = clean.clone()
    momentum = torch.zeros_like(clean)
    alpha = epsilon / steps
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        mean_gradient = cwt_gradient(model, cwt, x_adv, labels, copy_chunk)
        x_adv, momentum = _momentum_step(
            clean, x_adv, mean_gradient, momentum, epsilon, alpha, mu
        )
    return x_adv.detach()


@torch.inference_mode()
def attack_metrics(
    target_model: nn.Module,
    clean: Tensor,
    adversarial: Tensor,
    labels: Tensor,
) -> dict[str, int | float]:
    """Evaluate an untargeted attack only on clean-correct examples."""

    if clean.shape != adversarial.shape:
        raise ValueError("clean and adversarial shapes differ")
    clean_prediction = target_model(clean).argmax(1)
    adversarial_prediction = target_model(adversarial).argmax(1)
    eligible = clean_prediction.eq(labels)
    successful = eligible & adversarial_prediction.ne(labels)
    eligible_count = int(eligible.sum().item())
    successful_count = int(successful.sum().item())
    return {
        "eligible": eligible_count,
        "successful": successful_count,
        "attack_success_rate": successful_count / max(eligible_count, 1),
        "robust_accuracy": float(
            (adversarial_prediction[eligible] == labels[eligible]).float().mean().item()
        )
        if eligible_count
        else 0.0,
    }


__all__ = [
    "CWT",
    "CWTConfig",
    "attack_metrics",
    "cwt_gradient",
    "cwt_mifgsm",
    "mifgsm",
]
