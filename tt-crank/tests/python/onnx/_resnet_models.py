# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ResNet models shared by the onnx model tests and benchmarks."""

import torch
import transformers


class Logits(torch.nn.Module):
    """Unwrap a HF classifier's output to its logits tensor."""

    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x).logits


class Bf16Logits(Logits):
    """Native bf16 like torch's `model.to(bfloat16)`: bf16 in, bf16 logits out."""

    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__(m.to(torch.bfloat16))  # in place; callers pass a deepcopy


def hf_resnet50() -> torch.nn.Module:
    """Pretrained microsoft/resnet-50 (downloaded on first use)."""
    return transformers.ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50"
    ).eval()
