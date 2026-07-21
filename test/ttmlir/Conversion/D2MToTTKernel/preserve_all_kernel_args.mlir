// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s --check-prefix=PRUNE
// RUN: ttmlir-opt --ttcore-register-device "--convert-d2m-to-ttkernel=preserve-all-kernel-args=true" %s | FileCheck %s --check-prefix=PRESERVE

#l1 = #ttcore.memory_space<l1>

module {
  func.func @parent(%scalar0: i32, %scalar1: i32) {
    %cb = memref.alloc() {address = 112416 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    %buf = memref.alloc() {address = 104224 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @datamovement_kernel0, dm_core = 0>, #d2m.thread<datamovement, @datamovement_kernel1, dm_core = 1>]}
        ins(%buf : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%cb : memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
        additionalArgs(%scalar0, %scalar1 : i32, i32)
    return
  }

  // PRUNE-LABEL: func.func private @datamovement_kernel0
  // PRUNE-SAME: ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = scalar, operand_index = 2>]
  // PRESERVE-LABEL: func.func private @datamovement_kernel0
  // PRESERVE-SAME: ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = buffer_address, operand_index = 0>, <arg_type = scalar, operand_index = 2>, <arg_type = scalar, operand_index = 3>]
  func.func private @datamovement_kernel0() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 0>, tt.function_type = "kernel"} {
    %cb = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    %scalar0 = d2m.get_arg(2) : i32
    return
  }

  // PRUNE-LABEL: func.func private @datamovement_kernel1
  // PRUNE-SAME: ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = scalar, operand_index = 3>]
  // PRESERVE-LABEL: func.func private @datamovement_kernel1
  // PRESERVE-SAME: ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = buffer_address, operand_index = 0>, <arg_type = scalar, operand_index = 2>, <arg_type = scalar, operand_index = 3>]
  func.func private @datamovement_kernel1() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 1>, tt.function_type = "kernel"} {
    %buf = d2m.get_arg(0) : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %scalar1 = d2m.get_arg(3) : i32
    return
  }
}
