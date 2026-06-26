// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s --check-prefix=SMART
// RUN: ttmlir-opt --ttcore-register-device "--convert-d2m-to-ttkernel=force-compile-time-args=true" %s | FileCheck %s --check-prefix=FORCE

#l1 = #ttcore.memory_space<l1>

module {
  // SMART-LABEL: func.func private @arg_placement
  // SMART-SAME: ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = buffer_address, operand_index = 1>, <arg_type = local_semaphore, operand_index = 3>, <arg_type = scalar, operand_index = 2>]
  // SMART: ttkernel.get_common_arg_val
  // SMART-NOT: ttkernel.get_compile_time_arg_val
  // FORCE-LABEL: func.func private @arg_placement
  // FORCE-SAME: ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = buffer_address, operand_index = 1>, <arg_type = local_semaphore, operand_index = 3>, <arg_type = scalar, operand_index = 2>]
  // FORCE: ttkernel.get_compile_time_arg_val
  // FORCE-NOT: ttkernel.get_common_arg_val
  func.func private @arg_placement() attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
    %cb = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    %buf = d2m.get_arg(1) : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>
    %sem = d2m.get_arg(3) : !d2m.local_semaphore
    %scalar = d2m.get_arg(2) : ui32
    return
  }
}
