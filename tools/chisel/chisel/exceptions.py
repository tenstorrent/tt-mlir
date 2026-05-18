# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


class GoldenNotImplementedError(NotImplementedError):
    """Raised by execute_golden when no golden is registered for an op."""

    def __init__(self, op):
        super().__init__(f"{op.name}: no golden registered for {type(op).__name__}")
        self.op = op
