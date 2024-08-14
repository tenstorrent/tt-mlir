# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
from __future__ import annotations

# from build.python_packages.ttmlir.ir import *
# from build.python_packages.ttmlir.dialects import tt, ttir

from typing import List, Dict

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir


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

    def getRankedTensor(self, shape: List[int], data_type: "tt.DataType"):
        return RankedTensorType.get(shape, Type.parse(str(data_type), self.ctx), loc=self.cursor)

    def getCoreRangeAttr(self, offset: List[int], size: List[int]) -> "tt.ir.CoreRangeAttr":
        return tt.ir.CoreRangeAttr.get(self.ctx, offset, size)

    def getCircularBufferAttributesAttr(
        self, cb_id: int, core_range: "tt.ir.CoreRangeAttr", total_size: int, page_size: int, data_format: int
    ) -> "tt.ir.CircularBufferAttributesAttr":
        return tt.ir.CircularBufferAttributesAttr.get(self.ctx, cb_id, core_range, total_size, page_size, data_format)

    def getDataMovementConfigAttr(
        self, data_movement_type: int, compile_args: List[int]
    ) -> "tt.ir.DataMovementConfigAttr":
        return tt.ir.DataMovementConfigAttr.get(self.ctx, data_movement_type, compile_args)

    def getDataMovementAttributesAttr(
        self, core_range: "tt.ir.CoreRangeAttr", kernel_path: str, data_movement_config: "tt.ir.DataMovementConfigAttr"
    ) -> "tt.ir.DataMovementAttributesAttr":
        return tt.ir.DataMovementAttributesAttr.get(self.ctx, core_range, kernel_path, data_movement_config)

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
        self, core_range: "tt.ir.CoreRangeAttr", kernel_path: str, compute_config: "tt.ir.ComputeConfigAttr"
    ) -> "tt.ir.ComputeAttributesAttr":
        return tt.ir.ComputeAttributesAttr.get(self.ctx, core_range, kernel_path, compute_config)

    def getExternalGenericOp(
        self,
        inputs: List[Tensor],
        outputs: List[Tensor],
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
        return self.types_builder.getRankedTensor([64, 128], tt.DataType.Float32)

    def create_dummy_external_generic_op():
        return ttir.ExternalGenericOp()

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

        reader_config = self.types_builder.getDataMovementConfigAttr(tt.DataMovementType.Reader.value, [1, 1])
        writer_config = self.types_builder.getDataMovementConfigAttr(tt.DataMovementType.Writer.value, [1, 1])

        return [
            self.types_builder.getDataMovementAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
                reader_config,
            ),
            self.types_builder.getDataMovementAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                writer_config,
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

        return [
            self.types_builder.getComputeAttributesAttr(
                all_cores,
                "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
                compute_config,
            )
        ]


if __name__ == "__main__":
    builder = GenericOpBuilder()

    circular_buffer_attributes = builder.create_dummy_circular_buffer_attributes()
    data_movement_attributes = builder.create_dummy_data_movement_attributes()
    compute_attributes = builder.create_dummy_compute_attributes()

    in0 = builder.create_dummy_tensor()
    in1 = builder.create_dummy_tensor()
    out0 = builder.create_dummy_tensor()

    # Create the operation
    operation = ttir.external_generic(
        results_=[out0],  # Results of the operation
        inputs=[in0, in1],  # Inputs to the operation
        outputs=[out0],  # Outputs of the operation
        circular_buffer_attributes=circular_buffer_attributes,
        data_movement_attributes=data_movement_attributes,
        compute_attributes=compute_attributes,
        operand_constraints=None,
        loc=None,  # Optional location
        ip=None,  # Optional insertion point
    )

    # Example usage: print or further manipulate the created operation
    print(operation)
