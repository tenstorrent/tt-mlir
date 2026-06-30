// RUN: ttmlir-opt --split-input-file --ttcore-register-device --convert-d2m-to-ttkernel --convert-d2m-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Spatial lowering regression tests for D2M -> TTMetal.
// This file validates independent concerns with separate test funcs:
// 1) Region enqueue_programs merge into one ttmetal.enqueue_program.
// 2) GlobalSemaphore arg index remap during spatial merge.
// 3) Non-origin core-range derivation from d2m.generic grid maps.
// 4) BufferAddress arg index remap.
// 5) CBPort remap into merged cbs list (and sequential hardware cb_ports).
// 6) LocalSemaphore arg index remap during spatial merge.
// 7) Handle input of the spatial when remove view_layout.

// Single-region merge smoke test.
#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_single_region_merge
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK-NOT: d2m.spatial
  func.func @spatial_single_region_merge(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1> {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1100} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %cb_0 = memref.alloc() {address = 0x4000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_1 = memref.alloc() {address = 0x4100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>]}
        ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_single, dm_core = 1>, #d2m.thread<compute, @cp_single>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_0, %cb_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
    }
    return %out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_single() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_single() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_global_semaphore_arg_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <global_semaphore[2]>]>, dm_core = 1, noc0>
  // CHECK-SAME: #ttmetal.compute_config<@cp_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <global_semaphore[2]>]>, hifi4, true, false, false, [default]>
  // CHECK-SAME: #ttmetal.noc_config<@dm_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>, <global_semaphore[5]>]>, dm_core = 1, noc0>
  // CHECK-SAME: #ttmetal.compute_config<@cp_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>, <global_semaphore[5]>]>, hifi4, true, false, false, [default]>]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_global_semaphore_arg_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2000, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %sem_backing0 = memref.alloc() {alignment = 16 : i64, address = 0x3000} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    %sem_backing1 = memref.alloc() {alignment = 16 : i64, address = 0x3010} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    %sem0 = d2m.create_global_semaphore(%sem_backing0) {value = 0 : ui32} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1> -> !d2m.global_semaphore
    %sem1 = d2m.create_global_semaphore(%sem_backing1) {value = 0 : ui32} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1> -> !d2m.global_semaphore
    %cb_r0_0 = memref.alloc() {address = 0x4000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_r0_1 = memref.alloc() {address = 0x4100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_r1_0 = memref.alloc() {address = 0x4200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_r1_1 = memref.alloc() {address = 0x4300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>

    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r0, dm_core = 1>, #d2m.thread<compute, @cp_r0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%sem0, %cb_r0_0, %cb_r0_1 : !d2m.global_semaphore, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r1, dm_core = 1>, #d2m.thread<compute, @cp_r1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%sem1, %cb_r1_0, %cb_r1_1 : !d2m.global_semaphore, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }

    d2m.reset_global_semaphore(%sem0) {value = 0 : ui32} : !d2m.global_semaphore
    d2m.reset_global_semaphore(%sem1) {value = 0 : ui32} : !d2m.global_semaphore
    memref.dealloc %sem_backing1 : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    memref.dealloc %sem_backing0 : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

// Two-region non-origin core range from generic grid map.
#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_non_origin_core_range_from_grid_map
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_core_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, dm_core = 1, noc0>
  // CHECK-SAME: #ttmetal.compute_config<@cp_core_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, true, false, false, [default]>
  // CHECK-SAME: #ttmetal.noc_config<@dm_core_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>]>, dm_core = 1, noc0>
  // CHECK-SAME: #ttmetal.compute_config<@cp_core_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>]>, hifi4, true, false, false, [default]>]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_non_origin_core_range_from_grid_map(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1400} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2400, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %cb_core_r0_0 = memref.alloc() {address = 0x7000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_core_r0_1 = memref.alloc() {address = 0x7100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_core_r1_0 = memref.alloc() {address = 0x7200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_core_r1_1 = memref.alloc() {address = 0x7300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 2)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_core_r0, dm_core = 1>, #d2m.thread<compute, @cp_core_r0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_core_r0_0, %cb_core_r0_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_core_r1, dm_core = 1>, #d2m.thread<compute, @cp_core_r1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_core_r1_0, %cb_core_r1_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_core_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_core_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_core_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_core_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

// Two-region BufferAddress argsOffset remap test.
#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_buffer_address_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_b0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args<common_rt_args = [<buffer_address[0]>]  ct_args = [<cb_port[0]>, <cb_port[1]>]>, dm_core = 1, noc0>
  // CHECK-SAME: #ttmetal.noc_config<@dm_b1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args<common_rt_args = [<buffer_address[2]>]  ct_args = [<cb_port[2]>, <cb_port[3]>]>, dm_core = 1, noc0>]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_buffer_address_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1200} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2200, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %cb_b0_0 = memref.alloc() {address = 0x5000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_b0_1 = memref.alloc() {address = 0x5100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_b1_0 = memref.alloc() {address = 0x5200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_b1_1 = memref.alloc() {address = 0x5300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_b0, dm_core = 1>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_b0_0, %cb_b0_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_b1, dm_core = 1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_b1_0, %cb_b1_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_b0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_b1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// Two-region CBPort index remap: second region kernels must reference merged
// cbs slots [2,3], not [0,1]. Hardware cb_ports are 0,1,2,3 after merge remap.
#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_cb_port_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: cb_ports = array<i64: 0, 1, 2, 3>
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_cbport_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, dm_core = 1, noc0>
  // CHECK-SAME: #ttmetal.noc_config<@dm_cbport_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>]>, dm_core = 1, noc0>]
  // CHECK: operandSegmentSizes = array<i32: 4, 4>
  // CHECK-NOT: @dm_cbport_r1{{.*}}cb_port[0]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_cb_port_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1300} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2300, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %cb_cbp_r0_0 = memref.alloc() {address = 0x6000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_cbp_r0_1 = memref.alloc() {address = 0x6100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_cbp_r1_0 = memref.alloc() {address = 0x6200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_cbp_r1_1 = memref.alloc() {address = 0x6300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_cbport_r0, dm_core = 1>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_cbp_r0_0, %cb_cbp_r0_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_cbport_r1, dm_core = 1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%cb_cbp_r1_0, %cb_cbp_r1_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_cbport_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_cbport_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// Two-region LocalSemaphore arg index remap test.
#l1 = #ttcore.memory_space<l1>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_local_semaphore_arg_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: kernelConfigs = [
  // CHECK-DAG: @dm_ls_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <local_semaphore[2]>, <local_semaphore[3]>]>
  // CHECK-DAG: @cp_ls_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <local_semaphore[2]>, <local_semaphore[3]>]>
  // CHECK-DAG: @dm_ls_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>, <local_semaphore[6]>, <local_semaphore[7]>]>
  // CHECK-DAG: @cp_ls_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>, <local_semaphore[6]>, <local_semaphore[7]>]>
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_local_semaphore_arg_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1500} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2500, d2m.virtualGridInverseMapping = #vgm_11_inv, d2m.virtualGridForwardMapping = #vgm_11_fwd} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %ls_r0_0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %ls_r0_1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %ls_r1_0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %ls_r1_1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %cb_ls_r0_0 = memref.alloc() {address = 0x8000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_ls_r0_1 = memref.alloc() {address = 0x8100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_ls_r1_0 = memref.alloc() {address = 0x8200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>
    %cb_ls_r1_1 = memref.alloc() {address = 0x8300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>

    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_ls_r0, dm_core = 1>, #d2m.thread<compute, @cp_ls_r0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%ls_r0_0, %ls_r0_1, %cb_ls_r0_0, %cb_ls_r0_1 : !d2m.local_semaphore, !d2m.local_semaphore, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_ls_r1, dm_core = 1>, #d2m.thread<compute, @cp_ls_r1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%ls_r1_0, %ls_r1_1, %cb_ls_r1_0, %cb_ls_r1_1 : !d2m.local_semaphore, !d2m.local_semaphore, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1>)
      }

    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_ls_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>, <arg_type = local_semaphore, operand_index = 2>, <arg_type = local_semaphore, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_ls_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>, <arg_type = local_semaphore, operand_index = 2>, <arg_type = local_semaphore, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_ls_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>, <arg_type = local_semaphore, operand_index = 2>, <arg_type = local_semaphore, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_ls_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>, <arg_type = local_semaphore, operand_index = 2>, <arg_type = local_semaphore, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

// Regression: d2m.view_layout feeding d2m.spatial ins/outs should not leave
// unresolved unrealized_conversion_cast after d2m.generic bypasses view types.
#l1 = #ttcore.memory_space<l1>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_view_layout_inout_type_bridge
  // CHECK: "ttmetal.enqueue_program"
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK-NOT: d2m.view_layout
  func.func @spatial_view_layout_inout_type_bridge(
      %arg0: memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1>)
      -> memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1> {
    %view = d2m.view_layout %arg0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1> -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %out = memref.alloc() {alignment = 64 : i64, address = 0x2400} : memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1>
    %out_view = d2m.view_layout %out remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1> -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>]}
        ins(%view : memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%out_view : memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_spatial_view, dm_core = 1>]}
            ins(%view : memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
            outs(%out_view : memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
    }
    return %out : memref<4x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1>
  }

  func.func private @dm_spatial_view() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< >, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}
