import random

import pytest
import torch
from torch import nn

from src.attacks.cwt import CWT, CWTConfig, attack_metrics, cwt_mifgsm, mifgsm


class TinyModel(nn.Module):
    def __init__(self, channels=3, classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, classes),
        )

    def forward(self, x):
        return self.network(x)


def make_batch():
    torch.manual_seed(7)
    clean = torch.rand(2, 3, 16, 16)
    labels = torch.tensor([0, 1])
    return clean, labels


def assert_attack_invariants(clean, adversarial, epsilon):
    assert adversarial.shape == clean.shape
    assert torch.isfinite(adversarial).all()
    assert adversarial.min() >= 0
    assert adversarial.max() <= 1
    assert (adversarial - clean).abs().amax() <= epsilon + 1e-6


def test_config_validation():
    with pytest.raises(ValueError):
        CWT(CWTConfig(grid=0))
    with pytest.raises(ValueError):
        CWT(CWTConfig(copies=0))
    with pytest.raises(ValueError):
        CWT(CWTConfig(grid=2, rotated_blocks=5))
    with pytest.raises(ValueError):
        CWT(CWTConfig(scale_min=0.9))


def test_cwt_shape_and_gradient_flow():
    clean, _ = make_batch()
    clean.requires_grad_(True)
    transform = CWT(CWTConfig(copies=3))
    transformed = transform(clean)
    assert transformed.shape == (6, 3, 16, 16)
    gradient = torch.autograd.grad(transformed.square().mean(), clean)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_cwt_supports_grayscale_models():
    torch.manual_seed(9)
    clean = torch.rand(2, 1, 16, 16)
    labels = torch.tensor([0, 1])
    model = TinyModel(channels=1).eval().requires_grad_(False)
    transform = CWT(CWTConfig(copies=2, rotated_blocks=1))
    transformed = transform(clean)
    assert transformed.shape == (4, 1, 16, 16)
    adversarial = cwt_mifgsm(
        model, transform, clean, labels, epsilon=4 / 255, steps=2, copy_chunk=1
    )
    assert_attack_invariants(clean, adversarial, 4 / 255)


def test_identity_configuration():
    clean, _ = make_batch()
    transform = CWT(
        CWTConfig(
            grid=2,
            copies=1,
            scale_min=1.0,
            scale_max=1.0,
            max_angle=0.0,
            rotated_blocks=0,
        )
    )
    assert torch.equal(transform.forward_once(clean), clean)


@pytest.mark.parametrize("attack", [mifgsm, cwt_mifgsm])
def test_attack_invariants_and_frozen_model(attack):
    clean, labels = make_batch()
    model = TinyModel().eval().requires_grad_(False)
    epsilon = 4 / 255
    if attack is cwt_mifgsm:
        transform = CWT(CWTConfig(copies=2, rotated_blocks=1))
        adversarial = attack(
            model, transform, clean, labels, epsilon=epsilon, steps=2, copy_chunk=1
        )
    else:
        adversarial = attack(model, clean, labels, epsilon=epsilon, steps=2)
    assert_attack_invariants(clean, adversarial, epsilon)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_microbatching_matches_full_batch_with_fixed_randomness():
    clean, labels = make_batch()
    model = TinyModel().eval().requires_grad_(False)
    transform = CWT(CWTConfig(copies=4, rotated_blocks=2))

    random.seed(123)
    torch.manual_seed(123)
    chunked = cwt_mifgsm(
        model, transform, clean, labels, epsilon=4 / 255, steps=2, copy_chunk=1
    )
    random.seed(123)
    torch.manual_seed(123)
    full = cwt_mifgsm(
        model, transform, clean, labels, epsilon=4 / 255, steps=2, copy_chunk=4
    )
    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-5)


def test_attack_metrics_use_only_clean_correct_examples():
    class ThresholdModel(nn.Module):
        def forward(self, x):
            score = x.flatten(1).mean(1)
            return torch.stack((1 - score, score), dim=1)

    model = ThresholdModel()
    clean = torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4)))
    labels = torch.tensor([0, 0])
    adversarial = clean.clone()
    adversarial[0] = 1
    metrics = attack_metrics(model, clean, adversarial, labels)
    assert metrics["eligible"] == 1
    assert metrics["successful"] == 1
    assert metrics["attack_success_rate"] == 1.0
    assert metrics["robust_accuracy"] == 0.0
