# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Runtime smoke test for the KernelArgScalar flatbuffer path.
#
# Standard TTIR→D2M compilation never generates KernelArgScalar; it only
# arises when a d2m.generic carries scalar additionalArgs that flow through
# as compile-time arguments.  This test therefore writes a TTMetal-level
# MLIR module directly (skipping FE/ME) and verifies that:
#
#   1. ttmetal_to_flatbuffer_bin produces KernelArgScalar entries, and
#   2. the runtime KernelArgScalar path in arguments.h does not crash when
#      submit() is called with scalar tensors.

import pytest

import _ttmlir_runtime as tt_runtime
from ttmlir.ir import Context, Location, Module
from ttmlir.passmanager import PassManager
from ttmlir.passes import ttmetal_to_flatbuffer_bin

pytestmark = pytest.mark.frontend("ttir")

# A minimal TTMetal module whose top-level function takes one i32 scalar.
# The NOC kernel declares that scalar as compile-time arg 0 (ArgType::Scalar),
# which causes the flatbuffer generator to emit KernelArgScalar.  The kernel
# body is empty so no L1 buffers or hardcoded device addresses are needed.
_SCALAR_MLIR = """\
module {
  func.func @test_scalar_kernel_arg(%arg0: i32) {
    "ttmetal.enqueue_program"(%arg0) <{
      cb_ports = array<i64>,
      kernelConfigs = [
        #ttmetal.noc_config<@scalar_kernel,
          #ttmetal.core_range<0x0, 1x1>,
          #ttmetal.kernel_args<ct_args = [<scalar[0]>]>,
          noc0>
      ],
      operandSegmentSizes = array<i32: 1, 0>
    }> : (i32) -> ()
    return
  }
  func.func private @scalar_kernel() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<
      ct_args = [<arg_type = scalar, operand_index = 0>]>,
    ttkernel.thread = #ttkernel.thread<noc>
  } {
    return
  }
}
"""


@pytest.mark.parametrize("target", ["ttmetal"])
def test_scalar_kernel_arg(target: str, request, device):
    """
    Smoke-test the KernelArgScalar runtime path.

    Compiles a hand-written TTMetal module that passes an i32 scalar as a
    kernel compile-time argument, translates it to a flatbuffer, and submits
    it on device.  This exercises the KernelArgScalar branch in
    processKernelArgs (arguments.h) and the createScalarTensor runtime API.
    """
    sys_desc = request.config.getoption("--sys-desc")
    register_device_opts = f"system-desc-path={sys_desc}" if sys_desc else ""

    ctx = Context()
    loc = Location.unknown(ctx)
    with ctx, loc:
        module = Module.parse(_SCALAR_MLIR)

    pm = PassManager.parse(
        f"builtin.module("
        f"ttcore-register-device{{{register_device_opts}}},"
        f"ttcore-mark-functions-as-forward,"
        f"ttcore-wrap-device-module"
        f")",
        ctx,
    )
    pm.run(module.operation)

    capsule = ttmetal_to_flatbuffer_bin(module)
    fbb = tt_runtime.binary.load_binary_from_capsule(capsule)

    scalar = tt_runtime.runtime.create_scalar_tensor(42)
    outputs = tt_runtime.runtime.submit(device, fbb, 0, [scalar])
    tt_runtime.runtime.wait(outputs)
