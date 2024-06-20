# RUN: %python %s | FileCheck %s

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir

with Context() as ctx:
    ttir.register_dialect(ctx)

    module = Module.parse(
        """
    #any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = tensor.empty() : tensor<64x128xf32>
    %2 = tensor.empty() : tensor<64x128xf32>
    %3 = "ttir.multiply"(%0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    """
    )
    # CHECK: %[[C:.*]] = tensor.empty() : tensor<64x128xf32>
    # CHECK: %[[C:.*]] = "ttir.multiply"[[C:.*]] : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    print(str(module))
