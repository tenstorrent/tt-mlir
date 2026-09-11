# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared model definitions for benchmarks and forward-pass tests."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTLinear(nn.Module):
    def __init__(self, feat: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feat, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
