# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Dict

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir, arith, tensor, func


class CustomAttrsAndTypesBuilder:
    """
    Thin wrapper around Python bindings in `python/TTModule.cpp` which makes it easier to create Python objects by
    wrapping `::get` call and passing arguments to it. Methods are named like `getCamelCaseAttrName` to resemble MLIR
    CPP style. Arguments must match 1:1 with those used in bindings.

    TODO This, or something like this, should probably become a common builder all python tests will use and kept
    somewhere else.
    """

    def __init__(self):
        self.ctx = Context()
        tt.register_dialect(self.ctx)
        ttir.register_dialect(self.ctx)
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body

    def __init__(self, ctx: Context):
        self.ctx = ctx
        tt.register_dialect(self.ctx)
        ttir.register_dialect(self.ctx)
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body

    def getCoreRangeAttr(self, offset: List[int], size: List[int]) -> "tt.ir.CoreRangeAttr":
        return tt.ir.CoreRangeAttr.get(self.ctx, offset, size)

    def getCircularBufferAttributesAttr(
        self, cb_id: int, core_range: "tt.ir.CoreRangeAttr", total_size: int, page_size: int, data_format: int
    ) -> "tt.ir.CircularBufferAttributesAttr":
        return tt.ir.CircularBufferAttributesAttr.get(self.ctx, cb_id, core_range, total_size, page_size, data_format)

    def getDataMovementConfigAttr(
        self, data_movement_type: int, compile_args: List[int], defines: Dict[str, str]
    ) -> "tt.ir.DataMovementConfigAttr":
        return tt.ir.DataMovementConfigAttr.get(self.ctx, data_movement_type, compile_args, defines)

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

    def getComputeConfigAttr(
        self,
        math_fidelity: int,
        fp32_dest_acc_en: bool,
        preserve_fp32_precision: bool,
        math_approx_mode: bool,
        compile_args: List[int],
        defines: Dict[str, str],
    ) -> "tt.ir.ComputeConfigAttr":
        return tt.ir.ComputeConfigAttr.get(
            self.ctx, math_fidelity, fp32_dest_acc_en, preserve_fp32_precision, math_approx_mode, compile_args, defines
        )

    def getComputeAttributesAttr(
        self,
        core_range: "tt.ir.CoreRangeAttr",
        kernel_path: str,
        compute_config: "tt.ir.ComputeConfigAttr",
        runtime_args: List["tt.ir.RuntimeArgumentAttr"],
    ) -> "tt.ir.ComputeAttributesAttr":
        return tt.ir.ComputeAttributesAttr.get(self.ctx, core_range, kernel_path, compute_config, runtime_args)

    def getRuntimeArgumentAttr(
        self,
        ttnn_compute: bool,
        runtme_argument_type: int,
        val: int,
        argument_index: int,
        tensor_glob_id: int,
        core_range: "tt.ir.CoreRangeAttr",
    ) -> "tt.ir.RuntimeArgumentAttr":
        return tt.ir.RuntimeArgumentAttr.get(
            self.ctx, ttnn_compute, runtme_argument_type, val, argument_index, tensor_glob_id, core_range
        )

    def getExternalGenericOp(
        self,
        inputs: List,
        outputs: List,
        circular_buffer_attributes: List["tt.ir.CircularBufferAttributesAttr"],
        data_movement_attributes: List["tt.ir.DataMovementAttributesAttr"],
        compute_attributes: List["tt.ir.ComputeAttributesAttr"],
        operand_constraints=None,
    ):
        return ttir.ExternalGenericOp(
            None,
            inputs,
            outputs,
            circular_buffer_attributes,
            data_movement_attributes,
            compute_attributes,
            operand_constraints,
        )


class GenericOpBuilder:
    def __init__(self):
        self.ctx = Context()
        tt.register_dialect(self.ctx)
        ttir.register_dialect(self.ctx)
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.types_builder = CustomAttrsAndTypesBuilder(self.ctx)

    def create_dummy_tensor(self):
        shape = [64, 128]
        data_type = str(tt.DataType.Float32)

        with self.ctx, self.cursor:
            return tensor.EmptyOp(shape, Type.parse(data_type))

    def create_dummy_external_generic_op(self):
        # TODO Not working, I am confused, couldn't untangle it
        return None
        with self.ctx, self.cursor:
            input0 = self.create_dummy_tensor()
            input1 = self.create_dummy_tensor()
            output = self.create_dummy_tensor()

            shape = [64, 128]
            data_type = str(tt.DataType.Float32)
            operand_constraint = tt.OperandConstraint.AnyDevice
            return_type = RankedTensorType.get(shape, Type.parse(data_type))

            op = ttir.ExternalGenericOp(
                [return_type],
                [input0, input1],
                [output],
                circular_buffer_attributes[0],
                data_movement_attributes[0],
                compute_attributes[0],
                IntegerAttr.get(IntegerType.get_signless(32), operand_constraint.value),
            )

            print(op)
            return op

    def create_dummy_circular_buffer_attributes(self) -> List["tt.ir.CircularBufferAttributesAttr"]:
        all_cores = self.types_builder.getCoreRangeAttr([0, 0], [6, 6])

        data_format = tt.DataType.BFloat16

        page_size = 2048
        total_size = 2 * page_size

        return [
            self.types_builder.getCircularBufferAttributesAttr(
                tt.CB.c_in0.value, all_cores, total_size, page_size, data_format.value
            ),
            self.types_builder.getCircularBufferAttributesAttr(
                tt.CB.c_in1.value, all_cores, total_size, page_size, data_format.value
            ),
            self.types_builder.getCircularBufferAttributesAttr(
                tt.CB.c_out0.value, all_cores, total_size, page_size, data_format.value
            ),
        ]

    def create_dummy_data_movement_attributes(self) -> List["tt.ir.DataMovementAttributesAttr"]:
        all_cores = self.types_builder.getCoreRangeAttr([0, 0], [6, 6])

        reader_config = self.types_builder.getDataMovementConfigAttr(tt.DataMovementType.Reader.value, [1, 1], {})
        writer_config = self.types_builder.getDataMovementConfigAttr(tt.DataMovementType.Writer.value, [1, 1], {})

        return [
            self.types_builder.getDataMovementAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
                reader_config,
                [],
            ),
            self.types_builder.getDataMovementAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                writer_config,
                [],
            ),
        ]

    def create_dummy_compute_attributes(self) -> List["tt.ir.ComputeAttributesAttr"]:
        all_cores = self.types_builder.getCoreRangeAttr([0, 0], [6, 6])

        eltwise_add_defines = {
            "ELTWISE_OP": "add_tiles",
            "ELTWISE_OP_TYPE": "EltwiseBinaryType::ELWADD",
        }
        compute_config = self.types_builder.getComputeConfigAttr(
            tt.MathFidelity.HiFi4.value, False, False, False, [1, 1], eltwise_add_defines
        )
        runtime_args = self.types_builder.getRuntimeArgumentAttr(
            False, tt.RuntimeArgumentType.TensorAddr.value, 1, 1, 1, all_cores
        )

        return [
            self.types_builder.getComputeAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
                compute_config,
                [runtime_args],
            )
        ]


def get_constant(ctx, value: int):
    u32_ty = IntegerType.get_unsigned(32, ctx)
    result = arith.constant(u32_ty, value, loc=Location.unknown(ctx))
    return result


if __name__ == "__main__":
    builder = GenericOpBuilder()

    dummy_tensor = builder.create_dummy_tensor()
    circular_buffer_attributes = builder.create_dummy_circular_buffer_attributes()
    data_movement_attributes = builder.create_dummy_data_movement_attributes()
    compute_attributes = builder.create_dummy_compute_attributes()
    # ext = builder.create_dummy_external_generic_op()

    print(dummy_tensor)
    print(circular_buffer_attributes)
    print(data_movement_attributes)
    print(compute_attributes)
