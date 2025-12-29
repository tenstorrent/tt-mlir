# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

# Tests for the ttkernel::ArgSpecAttr Python bindings

from ttmlir.ir import *
from ttmlir.dialects import ttkernel

with Context() as ctx, Location.unknown():
    # Create ArgAttr instances for rt_args and ct_args.
    # ArgAttr.get() returns typed object directly.
    rt_arg1 = ttkernel.ir.ArgAttr.get(ctx._CAPIPtr, 0, 0, True)
    rt_arg2 = ttkernel.ir.ArgAttr.get(ctx._CAPIPtr, 0, 1, False)
    ct_arg1 = ttkernel.ir.ArgAttr.get(ctx._CAPIPtr, 1, 2, True)

    # Verify ArgAttr.get() returns typed object directly.
    # CHECK: rt_arg1 operand_index: 0
    print(f"rt_arg1 operand_index: {rt_arg1.operand_index}")
    # CHECK: rt_arg1 is_uniform: True
    print(f"rt_arg1 is_uniform: {rt_arg1.is_uniform}")

    # ArgSpecAttr.get() returns MlirAttribute for MLIR interop.
    spec_attr = ttkernel.ir.ArgSpecAttr.get(ctx._CAPIPtr, [rt_arg1, rt_arg2], [ct_arg1])

    # Downcast to access typed properties.
    spec = ttkernel.ir.ArgSpecAttr.maybe_downcast(spec_attr)

    # Test rt_args property returns typed ArgAttr directly.
    rt_args = spec.rt_args
    # CHECK: rt_args count: 2
    print(f"rt_args count: {len(rt_args)}")

    # Test ct_args property returns typed ArgAttr directly.
    ct_args = spec.ct_args
    # CHECK: ct_args count: 1
    print(f"ct_args count: {len(ct_args)}")

    # Verify returned ArgAttr items are typed directly.
    # CHECK: rt_arg1_back operand_index: 0
    print(f"rt_arg1_back operand_index: {rt_args[0].operand_index}")
    # CHECK: rt_arg2_back operand_index: 1
    print(f"rt_arg2_back operand_index: {rt_args[1].operand_index}")
    # CHECK: rt_arg2_back is_uniform: False
    print(f"rt_arg2_back is_uniform: {rt_args[1].is_uniform}")
    # CHECK: ct_arg1_back operand_index: 2
    print(f"ct_arg1_back operand_index: {ct_args[0].operand_index}")
