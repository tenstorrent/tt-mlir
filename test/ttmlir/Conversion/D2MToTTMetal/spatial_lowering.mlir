// RUN: ttmlir-opt --split-input-file --ttcore-register-device --convert-d2m-to-ttkernel --convert-d2m-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Spatial lowering regression test for D2M -> TTMetal.
// This test validates:
// 1) Two nested region programs are merged into one ttmetal.enqueue_program.
// 2) Per-region core ranges are preserved in merged kernel configs.
// 3) Index-based kernel args (global_semaphore operand index) are remapped
//    when args from multiple enqueues are concatenated.
// 4) CBPort operand_idx remaps into the merged `cbs` list per region; hardware
//    cb_ports are reassigned to sequential globally unique ids (temporary
//    workaround for spatial merge).

// Single-region merge smoke test.
#l1 = #ttcore.memory_space<l1>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_single_region_merge
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK-NOT: d2m.spatial
  func.func @spatial_single_region_merge(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1> {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1100} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>]}
        ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_single, noc = 0>, #d2m.thread<compute, @cp_single>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
    }
    return %out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_single() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_single() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

#l1 = #ttcore.memory_space<l1>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_global_semaphore_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <global_semaphore[2]>]>, noc0>
  // CHECK-SAME: #ttmetal.compute_config<@cp_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <global_semaphore[2]>]>, hifi4, true, false, false, [default]>
  // CHECK-SAME: #ttmetal.noc_config<@dm_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>, <global_semaphore[5]>]>, noc0>
  // CHECK-SAME: #ttmetal.compute_config<@cp_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>, <global_semaphore[5]>]>, hifi4, true, false, false, [default]>]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_global_semaphore_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2000} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %sem_backing0 = memref.alloc() {alignment = 16 : i64, address = 0x3000} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    %sem_backing1 = memref.alloc() {alignment = 16 : i64, address = 0x3010} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    %sem0 = d2m.create_global_semaphore(%sem_backing0) {value = 0 : ui32} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1> -> !d2m.global_semaphore
    %sem1 = d2m.create_global_semaphore(%sem_backing1) {value = 0 : ui32} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1> -> !d2m.global_semaphore

    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r0, noc = 0>, #d2m.thread<compute, @cp_r0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%sem0 : !d2m.global_semaphore)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r1, noc = 0>, #d2m.thread<compute, @cp_r1>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            additionalArgs(%sem1 : !d2m.global_semaphore)
      }

    d2m.reset_global_semaphore(%sem0) {value = 0 : ui32} : !d2m.global_semaphore
    d2m.reset_global_semaphore(%sem1) {value = 0 : ui32} : !d2m.global_semaphore
    memref.dealloc %sem_backing1 : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    memref.dealloc %sem_backing0 : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----

// Two-region BufferAddress argsOffset remap test.
#l1 = #ttcore.memory_space<l1>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_buffer_address_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_b0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args<rt_args = [<buffer_address[0]>] ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>
  // CHECK-SAME: #ttmetal.noc_config<@dm_b1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args<rt_args = [<buffer_address[2]>] ct_args = [<cb_port[2]>, <cb_port[3]>]>, noc0>]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_buffer_address_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1200} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2200} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_b0, noc = 0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_b1, noc = 0>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_b0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_b1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}

// -----

// Two-region CBPort index remap: second region kernels must reference merged
// cbs slots [2,3], not [0,1]. Hardware cb_ports are 0,1,2,3 after merge remap.
#l1 = #ttcore.memory_space<l1>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @spatial_two_regions_cb_port_remap
  // CHECK-COUNT-1: "ttmetal.enqueue_program"
  // CHECK: cb_ports = array<i64: 0, 1, 2, 3>
  // CHECK: kernelConfigs = [#ttmetal.noc_config<@dm_cbport_r0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>
  // CHECK-SAME: #ttmetal.noc_config<@dm_cbport_r1, #ttmetal.core_range<1x1, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[2]>, <cb_port[3]>]>, noc0>]
  // CHECK: operandSegmentSizes = array<i32: 4, 4>
  // CHECK-NOT: @dm_cbport_r1{{.*}}cb_port[0]
  // CHECK-NOT: d2m.spatial
  func.func @spatial_two_regions_cb_port_remap(
      %arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
      %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      -> (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>,
          memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    %out0 = memref.alloc() {alignment = 64 : i64, address = 0x1300} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %out1 = memref.alloc() {alignment = 64 : i64, address = 0x2300} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%arg0, %arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
        outs(%out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_cbport_r0, noc = 0>]}
            ins(%arg0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_cbport_r1, noc = 0>]}
            ins(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
            outs(%out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
      }
    return %out0, %out1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }

  func.func private @dm_cbport_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_cbport_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
}
