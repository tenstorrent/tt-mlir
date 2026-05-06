# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


class GoldenNotImplementedError(NotImplementedError):
    """Raised by execute_golden when no golden is registered for an op."""

    def __init__(self, op):
        op_name = op.name
        op_type_name = type(op).__name__
        super().__init__(f"{op_name}: no golden registered for {op_type_name}")
        self.op = op
        self.op_name = op_name
        self.op_type_name = op_type_name
