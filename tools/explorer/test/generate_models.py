# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
EMPTY = "%{} = tensor.empty() : tensor<1x64xf32>"
OP = '%{x} = "ttir.relu"(%{y}, %{z}) <{{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#tt.operand_constraint<dram|l1|scalar|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device|any_device_tile|l1_block_sharded>, #tt.operand_constraint<dram|l1|scalar|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device|any_device_tile|l1_block_sharded>]}}> : (tensor<1x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>'


for i in range(5000):
    print(EMPTY.format(i * 2))
    print(OP.format(x=i * 2 + 1, y=i * 2 - 1, z=i * 2))
