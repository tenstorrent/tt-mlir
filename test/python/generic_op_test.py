# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
from __future__ import annotations

from typing import List, Dict, Optional, Tuple

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir, tensor


class AttrsAndTypesBuilder:
    """
    Thin wrapper around Python classes auto generated from pybind definitions.

    This API aims to provide 1:1 type match with Cpp getters of tablegen-erated classes. Pybinds
    usually use integral types (such as ints for enums) and then use casts to reinterpret them as
    some Cpp type. This class circumvents that and provides getters with Python types used for
    arguments matching those in Cpp getters with Cpp types.

    Luckily, Python lists are matched with Cpp vectors, and dicts are matched with Cpp maps out of
    the box.

    `::mlir::MLIRContext*` argument is taken care of without being exposed in these getters.
    """

    def __init__(self, context: Optional[Context] = None):
        self.ctx = Context() if context is None else context

        self.loc = Location.unknown(self.ctx)
        self.module = Module.create(self.loc)
        self.insert_point = self.module.body

        tt.register_dialect(self.ctx)
        ttir.register_dialect(self.ctx)

    def getCoreRangeAttr(self, offset: List[int], size: List[int]) -> "tt.ir.CoreRangeAttr":
        return tt.ir.CoreRangeAttr.get(self.ctx, offset, size)

    def getOperandConstraintAttr(
        self, operand_constraint: "tt.OperandConstraint"
    ) -> "tt.ir.OperandConstraintAttr":
        return tt.ir.OperandConstraintAttr.get(self.ctx, operand_constraint.value)

    def getOperandConstraintArrayAttr(self, attrsArray: List["tt.ir.OperandConstraintAttr"]):
        return tt.ir.OperandConstraintAttr.get(self.ctx, attrsArray)

    def getCircularBufferAttributesAttr(
        self,
        cb: "tt.CB",
        core_range: "tt.ir.CoreRangeAttr",
        total_size: int,
        page_size: int,
        data_format: "tt.DataType",
    ) -> "tt.ir.CircularBufferAttributesAttr":
        return tt.ir.CircularBufferAttributesAttr.get(
            self.ctx, cb.value, core_range, total_size, page_size, data_format.value
        )

    def getCircularBufferAttributesArrayAttr(
        self, attrsArray: List["tt.ir.CircularBufferAttributesAttr"]
    ):
        return tt.ir.CircularBufferAttributesAttr.get(self.ctx, attrsArray)

    def getRuntimeArgumentAttr(
        self,
        ttnn_compute: bool,
        runtime_argument_type: "tt.RuntimeArgumentType",
        val: int,
        argument_index: int,
        tensor_glob_id: int,
        core_range: "tt.ir.CoreRangeAttr",
    ) -> "tt.ir.RuntimeArgumentAttr":
        return tt.ir.RuntimeArgumentAttr.get(
            self.ctx,
            ttnn_compute,
            runtime_argument_type.value,
            val,
            argument_index,
            tensor_glob_id,
            core_range,
        )

    def getDataMovementConfigAttr(
        self,
        data_movement_type: "tt.DataMovementType",
        compile_args: List[int],
        defines: Dict[str, str],
    ) -> "tt.ir.DataMovementConfigAttr":
        return tt.ir.DataMovementConfigAttr.get(
            self.ctx, data_movement_type.value, compile_args, defines
        )

    def getDataMovementAttributesAttr(
        self,
        core_range: "tt.ir.CoreRangeAttr",
        kernel_path: str,
        data_movement_config: "tt.ir.DataMovementConfigAttr",
        runtime_args: List["tt.ir.RuntimeArgumentAttr"],
    ) -> "tt.ir.DataMovementAttributesAttr":
        return tt.ir.DataMovementAttributesAttr.get(
            self.ctx, core_range, kernel_path, data_movement_config, runtime_args
        )

    def getDataMovementAttributesArrayAttr(
        self, attrsArray: List["tt.ir.DataMovementAttributesAttr"]
    ):
        return tt.ir.DataMovementAttributesAttr.get(self.ctx, attrsArray)

    def getComputeConfigAttr(
        self,
        math_fidelity: "tt.MathFidelity",
        fp32_dest_acc_en: bool,
        preserve_fp32_precision: bool,
        math_approx_mode: bool,
        compile_args: List[int],
        defines: Dict[str, str],
    ) -> "tt.ir.ComputeConfigAttr":
        return tt.ir.ComputeConfigAttr.get(
            self.ctx,
            math_fidelity.value,
            fp32_dest_acc_en,
            preserve_fp32_precision,
            math_approx_mode,
            compile_args,
            defines,
        )

    def getComputeAttributesAttr(
        self,
        core_range: "tt.ir.CoreRangeAttr",
        kernel_path: str,
        compute_config: "tt.ir.ComputeConfigAttr",
        runtime_args: List["tt.ir.RuntimeArgumentAttr"],
    ) -> "tt.ir.ComputeAttributesAttr":
        return tt.ir.ComputeAttributesAttr.get(
            self.ctx, core_range, kernel_path, compute_config, runtime_args
        )

    def getComputeAttributesArrayAttr(self, attrsArray: List["tt.ir.ComputeAttributesAttr"]):
        return tt.ir.ComputeAttributesAttr.get(self.ctx, attrsArray)

    def getExternalGenericOp(
        self,
        inputs: List[Value],
        outputs: List[Value],
        circular_buffer_attributes: List["tt.ir.CircularBufferAttributesAttr"],
        data_movement_attributes: List["tt.ir.DataMovementAttributesAttr"],
        compute_attributes: List["tt.ir.ComputeAttributesAttr"],
        operand_constraints: List["tt.ir.OperandConstraintAttr"],
    ):
        # Wraps list of attributes in an attribute before calling constructor.
        with self.ctx, self.loc:
            return ttir.ExternalGenericOp(
                [output.type for output in outputs],
                inputs,
                outputs,
                self.getCircularBufferAttributesArrayAttr(circular_buffer_attributes),
                self.getDataMovementAttributesArrayAttr(data_movement_attributes),
                self.getComputeAttributesArrayAttr(compute_attributes),
                self.getOperandConstraintArrayAttr(operand_constraints),
            )


class ExternalGenericOpBuilder:
    def __init__(self):
        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.types_builder = AttrsAndTypesBuilder(self.ctx)

    def create_empty_tensor(self):
        shape = [64, 128]
        data_type = tt.DataType.Float32

        with self.ctx, self.loc:
            return tensor.empty(shape, Type.parse(str(data_type)))

    def create_dummy_circular_buffer_attributes(self) -> List["tt.ir.CircularBufferAttributesAttr"]:
        all_cores = self.types_builder.getCoreRangeAttr([0, 0], [6, 6])
        data_format = tt.DataType.BFloat16
        page_size = 2048
        total_size = 2 * page_size

        return [
            self.types_builder.getCircularBufferAttributesAttr(
                tt.CB.c_in0, all_cores, total_size, page_size, data_format
            ),
            self.types_builder.getCircularBufferAttributesAttr(
                tt.CB.c_in1, all_cores, total_size, page_size, data_format
            ),
            self.types_builder.getCircularBufferAttributesAttr(
                tt.CB.c_out0, all_cores, total_size, page_size, data_format
            ),
        ]

    def create_dummy_data_movement_attributes(self) -> List["tt.ir.DataMovementAttributesAttr"]:
        all_cores = self.types_builder.getCoreRangeAttr([0, 0], [6, 6])

        reader_config = self.types_builder.getDataMovementConfigAttr(
            tt.DataMovementType.Reader, [1, 1], {}
        )
        writer_config = self.types_builder.getDataMovementConfigAttr(
            tt.DataMovementType.Writer, [1, 1], {}
        )

        runtime_args = self.types_builder.getRuntimeArgumentAttr(
            False, tt.RuntimeArgumentType.TensorAddr, 1, 1, 1, all_cores
        )

        return [
            self.types_builder.getDataMovementAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
                reader_config,
                [runtime_args],
            ),
            self.types_builder.getDataMovementAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                writer_config,
                [runtime_args],
            ),
        ]

    def create_dummy_compute_attributes(self) -> List["tt.ir.ComputeAttributesAttr"]:
        all_cores = self.types_builder.getCoreRangeAttr([0, 0], [6, 6])

        eltwise_add_defines = {
            "ELTWISE_OP": "add_tiles",
            "ELTWISE_OP_TYPE": "EltwiseBinaryType::ELWADD",
        }
        compute_config = self.types_builder.getComputeConfigAttr(
            tt.MathFidelity.HiFi4, False, False, False, [1, 1], eltwise_add_defines
        )
        runtime_args = self.types_builder.getRuntimeArgumentAttr(
            False, tt.RuntimeArgumentType.TensorAddr, 1, 1, 1, all_cores
        )

        return [
            self.types_builder.getComputeAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
                compute_config,
                [runtime_args],
            )
        ]

    def create_dummy_operand_constraints(
        self, total_num_of_input_and_output_operands: int
    ) -> List["tt.ir.OperandConstraintsAttr"]:
        operand_constraints = [
            tt.OperandConstraint.AnyDevice for i in range(total_num_of_input_and_output_operands)
        ]

        return [
            self.types_builder.getOperandConstraintAttr(constr) for constr in operand_constraints
        ]

    def create_dummy_external_generic_op(
        self,
        inputs: List[Value],
        outputs: List[Value],
        circular_buffer_attributes: List["tt.ir.CircularBufferAttributesAttr"],
        data_movement_attributes: List["tt.ir.DataMovementAttributesAttr"],
        compute_attributes: List["tt.ir.ComputeAttributesAttr"],
        operand_constraints: List["tt.ir.OperandConstraintAttr"],
    ):

        return self.types_builder.getExternalGenericOp(
            inputs,
            outputs,
            circular_buffer_attributes,
            data_movement_attributes,
            compute_attributes,
            operand_constraints,
        )


if __name__ == "__main__":
    builder = ExternalGenericOpBuilder()

    input0 = builder.create_empty_tensor()
    input1 = builder.create_empty_tensor()
    output = builder.create_empty_tensor()

    circular_buffer_attributes = builder.create_dummy_circular_buffer_attributes()
    data_movement_attributes = builder.create_dummy_data_movement_attributes()
    compute_attributes = builder.create_dummy_compute_attributes()
    operand_constraints = builder.create_dummy_operand_constraints(3)  # 2 inputs, 1 output

    ext = builder.create_dummy_external_generic_op(
        [input0, input1],
        [output],
        circular_buffer_attributes,
        data_movement_attributes,
        compute_attributes,
        operand_constraints,
    )

    print(ext)

    # Without explicitly deleting, python reports
    # `error: 'tensor.empty' op operation destroyed but still has uses`.
    del ext
