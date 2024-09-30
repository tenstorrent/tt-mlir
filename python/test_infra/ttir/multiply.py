# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional, List

from .ttir_op_test_base import TTIROpTestBase

from ttmlir.ir import Value, Context, Location, OpView
from ttmlir.dialects import ttir, tt


class Multiply(TTIROpTestBase):
    """Test harness for `ttir.multiply` op."""

    def __init__(self, ctx: Context, location: Location):
        super().__init__(ctx, location)

    @property
    def name(self) -> str:
        return "multiply"

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1

    def golden(self, in0: torch.Tensor, in1: torch.Tensor) -> torch.Tensor:
        return torch.multiply(in0, in1)

    def build(
        self,
        in_values: List[Value],
        out_values: List[Value],
        operand_constraints: Optional[List[tt.OperandConstraint]] = None,
    ) -> OpView:
        with self.ctx, self.loc:
            return ttir.MultiplyOp(
                [output.type for output in out_values],
                in_values,
                out_values,
                self._wrap_operand_constraints(
                    len(in_values) + len(out_values), operand_constraints
                ),
            )
