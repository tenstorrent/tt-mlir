// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel --convert-d2m-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
!slot = memref<1x1x!ttcore.tile<32x32, f32>, #l1>

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @spatial_global_cb_zip
  // CHECK: %[[GCB:.*]] = ttmetal.create_global_circular_buffer
  // CHECK-SAME: mapping = #d2m.global_cb_mapping<zip, sender = #ttcore.core_range<(0,0), (0,0)>, receiver = #ttcore.core_range<(1,1), (1,1)>>
  // CHECK-SAME: size = 8192
  // CHECK: "ttmetal.enqueue_program"({{.*}}%[[GCB]]{{.*}})
  // CHECK-NOT: d2m.spatial
  // CHECK-NOT: d2m.create_global_cb
  func.func @spatial_global_cb_zip(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2000, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %gcb = d2m.create_global_cb {
      mapping = #d2m.global_cb_mapping<zip,
        sender = #ttcore.core_range<(0, 0), (0, 0)>,
        receiver = #ttcore.core_range<(1, 1), (1, 1)>>,
      num_slots = 2 : i64
    } : !d2m.global_cb<!slot>

    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>,
                                #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
                           memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
                            memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r0, dm_core = 1>, #d2m.thread<compute, @cp_r0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%gcb : !d2m.global_cb<!slot>)
    }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r1, dm_core = 1>, #d2m.thread<compute, @cp_r1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%gcb : !d2m.global_cb<!slot>)
    }
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
                          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}
