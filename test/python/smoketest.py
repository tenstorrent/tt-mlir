# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

import ttmlir
from ttmlir.dialects import ttcore, ttir
from ttmlir.ir import *

with ttmlir.Context() as ctx:

    module = Module.parse(
        """
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = ttir.empty() : tensor<64x128xf32>
    %3 = "ttir.multiply"(%0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    """
    )
    # CHECK: = ttir.empty() : tensor<64x128xf32>
    # CHECK: = "ttir.multiply"
    # CHECK-SAME: (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    print(str(module))
