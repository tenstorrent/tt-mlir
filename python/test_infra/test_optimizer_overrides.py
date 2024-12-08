# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import ttmlir.optimizer_overrides as oo

from ttmlir.optimizer_overrides import MemoryLayoutAnalysisPolicyType
from ttmlir.optimizer_overrides import BufferType
from ttmlir.optimizer_overrides import Layout
from ttmlir.optimizer_overrides import TensorMemoryLayout
from ttmlir.optimizer_overrides import DataType


def main():

    print("\n\n ================ TESTING START ================ \n\n")

    # ----------------------------------------------------------------------------- #
    # Instantiate the OptimizerOverridesHandler
    # ----------------------------------------------------------------------------- #
    obj = oo.OptimizerOverridesHandler()

    # ----------------------------------------------------------------------------- #
    # Test setters and getters
    # ----------------------------------------------------------------------------- #

    # Enable Optimizer
    obj.set_enable_optimizer(True)
    print(f"Enable optimizer: {obj.get_enable_optimizer()}")
    obj.set_enable_optimizer(False)
    print(f"Enable optimizer: {obj.get_enable_optimizer()}")

    # Memory Reconfig
    obj.set_memory_reconfig(True)
    print(f"Memory Reconfig: {obj.get_memory_reconfig()}")
    obj.set_memory_reconfig(False)
    print(f"Memory Reconfig: {obj.get_memory_reconfig()}")

    # Enable Memory Layout Analysis
    obj.set_enable_memory_layout_analysis(True)
    print(f"Enable Memory Layout Analysis: {obj.get_enable_memory_layout_analysis()}")
    obj.set_enable_memory_layout_analysis(False)
    print(f"Enable Memory Layout Analysis: {obj.get_enable_memory_layout_analysis()}")

    # Enable Memory Layout Analysis Policy
    obj.set_enable_memory_layout_analysis_policy(True)
    print(
        f"Enable Memory Layout Analysis Policy: {obj.get_enable_memory_layout_analysis_policy()}"
    )
    obj.set_enable_memory_layout_analysis_policy(False)
    print(
        f"Enable Memory Layout Analysis Policy: {obj.get_enable_memory_layout_analysis_policy()}"
    )

    # Memory Layout Analysis Policy
    obj.set_memory_layout_analysis_policy(MemoryLayoutAnalysisPolicyType.DFSharding)
    print(f"Memory Layout Analysis Policy: {obj.get_memory_layout_analysis_policy()}")
    obj.set_memory_layout_analysis_policy(MemoryLayoutAnalysisPolicyType.L1Interleaved)
    print(f"Memory Layout Analysis Policy: {obj.get_memory_layout_analysis_policy()}")

    # System Descriptor Path
    obj.set_system_desc_path("System Descriptor Path")
    print(f"System Descriptor Path: {obj.get_system_desc_path()}")

    # Max Legal Layouts
    obj.set_max_legal_layouts(10)
    print(f"Max Legal Layouts: {obj.get_max_legal_layouts()}")

    # Mesh Shape
    obj.set_mesh_shape([1, 2, 3])
    print(f"Mesh Shape: {obj.get_mesh_shape()}")

    # ----------------------------------------------------------------------------- #
    # Test Input Layout and Output Layout
    # ----------------------------------------------------------------------------- #

    # Input Layout
    obj.add_input_layout_override("add", [1, 2])
    obj.add_input_layout_override("mul", [0, 1])
    obj.add_input_layout_override("sub", [2, 3])
    print(f"Input Layout: {obj.get_input_layout_overrides()}\n")

    # Output Layout
    obj.add_output_layout_override(
        "add",
        [0, 1],
        BufferType.DRAM,
        TensorMemoryLayout.HeightSharded,
        Layout.RowMajor,
        DataType.Float16,
    )
    obj.add_output_layout_override(
        "mul",
        [1, 2],
        BufferType.L1,
        TensorMemoryLayout.WidthSharded,
        Layout.RowMajor,
        DataType.BFloat16,
    )
    obj.add_output_layout_override(
        "sub",
        [2, 3],
        BufferType.SystemMemory,
        TensorMemoryLayout.BlockSharded,
        Layout.Tile,
        DataType.UInt16,
    )
    print(f"Output Layout: {obj.get_output_layout_overrides()}\n")

    # ----------------------------------------------------------------------------- #
    # Test string method
    # ----------------------------------------------------------------------------- #
    print(f"Optimizer Override string: {obj.to_string()}")

    print("\n\n ================ TESTING END ================ \n\n")


if __name__ == "__main__":
    main()
