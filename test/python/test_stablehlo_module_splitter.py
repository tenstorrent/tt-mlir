# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ttmlir.stablehlo_module_splitter import StableHLOModuleSplitter

if __name__ == "__main__":
    shlo_module_str = """
        module {
        func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
            %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
            %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
            %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
            return %2 : tensor<1x128xf32>
        }
        }
    """

    splitter: StableHLOModuleSplitter = StableHLOModuleSplitter.create_from_module_str(
        shlo_module_str
    )
    print(splitter.get_sub_ops())
    print([str(m) for m in splitter.get_sub_modules()])
