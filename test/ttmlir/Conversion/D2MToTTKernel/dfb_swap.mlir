// Verify that with `use-dfbs=true` the D2M->TTKernel conversion swaps
// d2m.wait / reserve / push / pop into the ttkernel.dfb_* family (and
// the kernel ArgSpec ct_args use ArgType::DFBId). Also verify that one
// ttkernel.dfb_finish is emitted per DFB at end of the kernel func.

// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel="use-dfbs=true" %s | FileCheck %s --check-prefix=DFB
// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s --check-prefix=CB

#l1_ = #ttcore.memory_space<l1>

module {
  // DFB-LABEL: func.func @kernel_passthrough
  // DFB-SAME: ct_args = [<arg_type = dfb_id, operand_index = 0>, <arg_type = dfb_id, operand_index = 1>]

  // CB-LABEL: func.func @kernel_passthrough
  // CB-SAME: ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]
  func.func @kernel_passthrough() attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    // DFB: ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.dfb<1, !ttcore.tile<32x32, f32>, 1, 1>
    // DFB: ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.dfb<1, !ttcore.tile<32x32, f32>, 1, 1>
    // DFB: ttkernel.dfb_wait_front
    // DFB: ttkernel.dfb_reserve_back
    // DFB: ttkernel.dfb_pop_front
    // DFB: ttkernel.dfb_push_back
    // DFB-NOT: ttkernel.cb_wait_front

    // CB: ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
    // CB: ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
    // CB: ttkernel.cb_wait_front
    // CB: ttkernel.cb_reserve_back
    // CB: ttkernel.cb_pop_front
    // CB: ttkernel.cb_push_back
    // CB-NOT: ttkernel.dfb_wait_front
    %in = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %t = memref.load %in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    memref.store %t, %out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>

    // Verify dfb_finish is emitted once per DFB at end of func.
    // DFB-COUNT-2: ttkernel.dfb_finish
    // DFB: return
    // CB-NOT: ttkernel.dfb_finish
    return
  }
}
