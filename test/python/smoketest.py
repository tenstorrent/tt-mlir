# RUN: %python %s | FileCheck %s

from mlir_ttmlir.ir import *
from mlir_ttmlir.dialects import builtin as builtin_d, ttmlir as ttmlir_d

with Context():
    ttmlir_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = ttmlir.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: ttmlir.foo %[[C]] : i32
    print(str(module))
