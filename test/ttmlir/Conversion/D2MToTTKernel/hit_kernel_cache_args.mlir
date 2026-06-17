// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s --check-prefix=DEFAULT
// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel="hit-kernel-cache=true" %s | FileCheck %s --check-prefix=HIT
// RUN: ttmlir-opt --ttcore-register-device --d2m-to-ttkernel-pre-emitc-pipeline="hit-kernel-cache=true" %s | FileCheck %s --check-prefix=PIPELINE

#l1 = #ttcore.memory_space<l1>

module {
  func.func private @kernel() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 1>} {
    %addr = d2m.get_arg(0) : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %scalar = d2m.get_arg(1) : i32
    %gsem = d2m.get_arg(2) : !d2m.global_semaphore
    %cb = d2m.get_arg(3) : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1>
    %lsem = d2m.get_arg(4) : !d2m.local_semaphore
    return
  }
}

// DEFAULT-LABEL: func.func private @kernel
// DEFAULT-SAME: ttkernel.arg_spec< ct_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = scalar, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>, <arg_type = cb_port, operand_index = 3>, <arg_type = local_semaphore, operand_index = 4>]>
// DEFAULT-COUNT-5: ttkernel.get_compile_time_arg_val
// DEFAULT-NOT: ttkernel.get_arg_val

// HIT: module attributes {{.*}}ttkernel.hit_kernel_cache
// HIT-LABEL: func.func private @kernel
// HIT-SAME: ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = scalar, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = local_semaphore, operand_index = 4>]>
// HIT-COUNT-3: ttkernel.get_arg_val
// HIT-COUNT-2: ttkernel.get_compile_time_arg_val

// PIPELINE: module attributes {{.*}}ttkernel.hit_kernel_cache
// PIPELINE-LABEL: func.func private @kernel
// PIPELINE-SAME: ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = scalar, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = local_semaphore, operand_index = 4>]>
