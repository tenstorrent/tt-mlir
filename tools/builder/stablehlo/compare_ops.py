# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare operations for StableHLO to TTIR conversion.
This module adds support for EQ, NE, GE, GT, LE, LT comparison operations.
"""

from typing import Optional, List
from ttmlir.ir import Operand, OpResult, UnitAttr, Location
from ttmlir.dialects import stablehlo, sdy

from builder.base.builder import Builder


class CompareOpsMixin:
    """Mixin class providing compare operations for StableHLOBuilder."""

    ############### stablehlo.CompareOp - Equal ###############

    def compare_eq(
        self: Builder,
        lhs: Operand,
        rhs: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """Element-wise equal comparison."""
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        compare_direction = stablehlo.ComparisonDirectionAttr.get("EQ", self._ctx)
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        op = stablehlo.CompareOp(
            lhs,
            rhs,
            compare_direction,
            compare_type=compare_type,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            import torch
            lhs_tensor = self._get_golden_tensor(lhs)
            rhs_tensor = self._get_golden_tensor(rhs)
            golden_output = torch.eq(lhs_tensor, rhs_tensor)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ############### stablehlo.CompareOp - NotEqual ###############

    def compare_ne(
        self: Builder,
        lhs: Operand,
        rhs: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """Element-wise not-equal comparison."""
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        compare_direction = stablehlo.ComparisonDirectionAttr.get("NE", self._ctx)
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        op = stablehlo.CompareOp(
            lhs,
            rhs,
            compare_direction,
            compare_type=compare_type,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            import torch
            lhs_tensor = self._get_golden_tensor(lhs)
            rhs_tensor = self._get_golden_tensor(rhs)
            golden_output = torch.ne(lhs_tensor, rhs_tensor)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ############### stablehlo.CompareOp - GreaterEqual ###############

    def compare_ge(
        self: Builder,
        lhs: Operand,
        rhs: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """Element-wise greater-than-or-equal comparison."""
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        compare_direction = stablehlo.ComparisonDirectionAttr.get("GE", self._ctx)
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        op = stablehlo.CompareOp(
            lhs,
            rhs,
            compare_direction,
            compare_type=compare_type,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            import torch
            lhs_tensor = self._get_golden_tensor(lhs)
            rhs_tensor = self._get_golden_tensor(rhs)
            golden_output = torch.ge(lhs_tensor, rhs_tensor)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ############### stablehlo.CompareOp - GreaterThan ###############

    def compare_gt(
        self: Builder,
        lhs: Operand,
        rhs: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """Element-wise greater-than comparison."""
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        compare_direction = stablehlo.ComparisonDirectionAttr.get("GT", self._ctx)
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        op = stablehlo.CompareOp(
            lhs,
            rhs,
            compare_direction,
            compare_type=compare_type,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            import torch
            lhs_tensor = self._get_golden_tensor(lhs)
            rhs_tensor = self._get_golden_tensor(rhs)
            golden_output = torch.gt(lhs_tensor, rhs_tensor)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ############### stablehlo.CompareOp - LessEqual ###############

    def compare_le(
        self: Builder,
        lhs: Operand,
        rhs: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """Element-wise less-than-or-equal comparison."""
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        compare_direction = stablehlo.ComparisonDirectionAttr.get("LE", self._ctx)
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        op = stablehlo.CompareOp(
            lhs,
            rhs,
            compare_direction,
            compare_type=compare_type,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            import torch
            lhs_tensor = self._get_golden_tensor(lhs)
            rhs_tensor = self._get_golden_tensor(rhs)
            golden_output = torch.le(lhs_tensor, rhs_tensor)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ############### stablehlo.CompareOp - LessThan ###############

    def compare_lt(
        self: Builder,
        lhs: Operand,
        rhs: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """Element-wise less-than comparison."""
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        compare_direction = stablehlo.ComparisonDirectionAttr.get("LT", self._ctx)
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        op = stablehlo.CompareOp(
            lhs,
            rhs,
            compare_direction,
            compare_type=compare_type,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            import torch
            lhs_tensor = self._get_golden_tensor(lhs)
            rhs_tensor = self._get_golden_tensor(rhs)
            golden_output = torch.lt(lhs_tensor, rhs_tensor)
            self._set_golden_tensor(op_result, golden_output)

        return op_result
