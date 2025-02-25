# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir

with Context() as ctx:
    ttir.register_dialect(ctx)

    module = Module.parse(
        """
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = tensor.empty() : tensor<64x128xf32>
    %2 = tensor.empty() : tensor<64x128xf32>
    %3 = "ttir.multiply"(%0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    """
    )
    # CHECK: %[[C:.*]] = tensor.empty() : tensor<64x128xf32>
    # CHECK: = "ttir.multiply" : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    print(str(module))
