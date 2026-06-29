// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

// Test that d2m.get_cb ops correctly lower to get_common_arg_val ops, making CB
// ports programmable via common runtime args. This is critical for cache reuse
// because CB port assignments do not need to participate in the kernel hash.

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @custom_cb_ports
  // Verify that operand indices appear in the ArgSpec rt_args.
  // CHECK-SAME: rt_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 5>]
  func.func @custom_cb_ports() attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %cb0 = d2m.get_cb(3) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %cb1 = d2m.get_cb(5) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    // CHECK: ttkernel.get_common_arg_val
    // CHECK: ttkernel.get_common_arg_val
    %in = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %t = memref.load %in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    memref.store %t, %out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @default_cb_ports
  // When port == operand_index (the common case), verify it still works.
  // CHECK-SAME: rt_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]
  func.func @default_cb_ports() attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    // CHECK: ttkernel.get_common_arg_val
    // CHECK: ttkernel.get_common_arg_val
    %in = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %t = memref.load %in[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    memref.store %t, %out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }
}
