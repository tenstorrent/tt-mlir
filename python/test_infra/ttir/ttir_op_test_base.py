# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod, ABC
from ttmlir.ir import Value, OpView, Context, Location
from ttmlir.dialects import tt, ttir
from torch import Tensor
from typing import Optional, List


class TTIROpTestBase(ABC):
    """
    Abstract base class from which tests for concrete ttir ops are derived.

    Implement provided abstract interface and TTIRBuilder will take care of the rest.
    """

    def __init__(self, ctx: Context, location: Location):
        self.ctx = ctx
        self.loc = location
        tt.register_dialect(self.ctx)
        ttir.register_dialect(self.ctx)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        pass

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        pass

    @abstractmethod
    def golden(self, in0: Tensor, in1: Tensor) -> Tensor:
        pass

    @abstractmethod
    def build(
        self,
        in_values: List[Value],
        out_values: List[Value],
        operand_constraint: Optional[tt.OperandConstraint] = None,
    ) -> OpView:
        pass

    def _wrap_operand_constraints(
        self,
        num_operands: int,
        operand_constraints: Optional[List[tt.OperandConstraint]] = None,
    ) -> tt.ir.OperandConstraintAttr:
        """
        Helper method to prepack operand constraints given as a list of enums to a list of
        tt.ir.OperandConstraintAttr and wraps that list in an tt.ir.OperandConstraintAttr.
        """
        operand_constraints = (
            operand_constraints
            if operand_constraints is not None
            else [tt.OperandConstraint.Any for _ in range(num_operands)]
        )

        return tt.ir.OperandConstraintAttr.get(
            self.ctx,
            [
                tt.ir.OperandConstraintAttr.get(self.ctx, operand_constraint)
                for operand_constraint in operand_constraints
            ],
        )
