// RUN: ttmlir-opt --split-input-file --ttcore-register-device="system-desc-path=%system_desc_path%" --convert-d2m-to-ttnn %s | FileCheck %s

// D2MSpatialRewriter tests: one merged ttnn.generic per spatial; tensor operands
// use kernel_arg_address_of_tensor with unified IO order (regions in sequence,
// first-seen findIOTensor dedupe). CB and kernel semaphore indices stay per-region
// local (may repeat across regions; core_ranges disambiguate).
// ttnn.empty shard_spec core_range per output.
// Each d2m.spatial region holds only d2m.generic; alloc/stream_layout/dealloc live outside.
//
// TC matrix:
// - single_region: 1 region, 2 in / 1 out, 3 CBs, baseline.
// - multi_inputs_not_shared: 2 regions, each 1 in + 1 out (no shared input); CB ct_args 0,1 per region; unified IO order in0,out0,in1,out1 -> address 0,1,2,3.
// - multi_cb_2_plus_3: 2 regions, r0: 2 CBs, r1: 3 CBs, shared in0; r1 uses local CB 0,1,2 in kernels; unified IO order in0,out0,in1,out1 with in0 deduped.
// - multi_region: 2 regions, 2 shared inputs, 2 outputs; each region ct_args CB 0,1,2; address 0,1,2,3 (last).
// - multi_region_semaphore: 2 regions; semaphore_at<0,1> per region; merged semaphores keep id 0,1 per region (listed twice).
// - two_global_semaphores: 2 regions; each generic has one additional global semaphore; merged common_rt_args remap flat operand_index 2 to unified slots 4 and 5.

// -----
// TC1: Single region.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram1 = #ttcore.memory_space<dram>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @spatial_single_region
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK: "ttnn.empty"(%{{.*}}) {{.*}}#ttnn.core_range<(0,0), (0,0)>{{.*}}tensor<64x64xf32, #ttnn_layout2>
  // CHECK: "ttnn.generic"(%arg0, %arg1, %{{.*}}) {{.*}}program = #ttnn.program<
  //
  //     TC1: single region, core (0,0), 3 CBs
  //   kernel   | core_ranges | ct_args | common_rt_args | rt_args
  //   ---------|-------------|---------|----------------|--------
  //   dm_s0    | (0,0)       | 0,1,2   | address<0>     | [*]
  //   dm_s1    | (0,0)       | 0,1,2   | address<1>     | [*]
  //   cp_s0    | (0,0)       | 0,1,2   | []             | [*]
  //
  // CHECK-SAME: {{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_s0{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>{{.*}}#ttnn.kernel_arg_address_of_tensor<0>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME: {{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_s1{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>{{.*}}#ttnn.kernel_arg_address_of_tensor<1>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME: {{.*}}#ttnn.compute_kernel<symbol_ref = @cp_s0{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>{{.*}}common_rt_args = [], rt_args = [{{[^]]*}}]>], cbs =
  //     cbs: kernel_cb 2; semaphores
  // CHECK-SAME: {{.*}}kernel_cb_global_buffer_address_of_tensor<2>{{.*}}semaphores = []
  func.func @spatial_single_region(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> tensor<64x64xf32, #ttnn_layout2> {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>
    %cast_1 = ttir.ttnn_metal_layout_cast %0 : tensor<64x64xf32, #ttnn_layout2> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %view_tc1_0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc1_1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %cb_tc1_0 = memref.alloc() {address = 0x4000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc1_1 = memref.alloc() {address = 0x4100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc1_2 = d2m.operand_alias %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>]}
        ins(%cast, %cast_0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>)
        outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_s0, noc = 0>, #d2m.thread<datamovement, @dm_s1, noc = 1>, #d2m.thread<compute, @cp_s0>]}
            ins(%view_tc1_0, %view_tc1_1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc1_0, %cb_tc1_1, %cb_tc1_2 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
    }
    %cast_2 = ttir.ttnn_metal_layout_cast %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout2>
    return %cast_2 : tensor<64x64xf32, #ttnn_layout2>
  }
  func.func private @dm_s0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_s1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_s0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----
// TC2: Multi-region, inputs NOT shared. Each region 1 in + 1 out. Global IOs: in0, in1, out0, out1. CB descriptors 2+2; ct_args CB 0,1 each region; address 0,1,2,3.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram1 = #ttcore.memory_space<dram>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @spatial_multi_inputs_not_shared
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK: "ttnn.generic"(%arg0, %{{.*}}, %arg1, %{{.*}}) {{.*}}program = #ttnn.program<
  //
  //     Region0 (core (0,0)): cb 0,1; address 0,1 (in0 then out0)
  //   kernel   | core_ranges | ct_args | common_rt_args | rt_args
  //   ---------|-------------|---------|----------------|--------
  //   dm_ns0   | (0,0)       | 0,1     | address<0>     | [*]
  //   dm_ns1   | (0,0)       | 0,1     | address<1>     | [*]
  //   cp_ns0   | (0,0)       | 0,1     | []             | [*]
  //
  // CHECK-SAME: {{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_ns0{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}#ttnn.kernel_arg_address_of_tensor<0>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_ns1{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}#ttnn.kernel_arg_address_of_tensor<1>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.compute_kernel<symbol_ref = @cp_ns0{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}common_rt_args = [], rt_args = [{{[^]]*}}]>,
  //
  //     Region1 (core (1,1)): local cb 0,1 (same ids as region0; different core); address 2,3
  //   kernel   | core_ranges | ct_args | common_rt_args | rt_args
  //   ---------|-------------|---------|----------------|--------
  //   dm_ns2   | (1,1)       | 0,1     | address<2>     | [*]
  //   dm_ns3   | (1,1)       | 0,1     | address<3>     | [*]
  //   cp_ns1   | (1,1)       | 0,1     | []             | [*]
  //
  // CHECK-SAME:{{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_ns2{{.*}}core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}#ttnn.kernel_arg_address_of_tensor<2>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_ns3{{.*}}core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}#ttnn.kernel_arg_address_of_tensor<3>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.compute_kernel<symbol_ref = @cp_ns1{{.*}}core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}common_rt_args = [], rt_args = [{{[^]]*}}]>], cbs =
  //
  //     cbs: kernel_cb 2, 3; semaphores
  // CHECK-SAME:{{.*}}kernel_cb_global_buffer_address_of_tensor<1>{{.*}}kernel_cb_global_buffer_address_of_tensor<3>{{.*}}semaphores = []
  func.func @spatial_multi_inputs_not_shared(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> (tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>) {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %1 = d2m.empty() : tensor<64x64xf32, #ttnn_layout3>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>
    %cast_1 = ttir.ttnn_metal_layout_cast %0 : tensor<64x64xf32, #ttnn_layout2> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %cast_2 = ttir.ttnn_metal_layout_cast %1 : tensor<64x64xf32, #ttnn_layout3> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %view_tc2_r0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc2_r1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %cb_tc2_r0_0 = memref.alloc() {address = 0x5000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc2_r0_1 = d2m.operand_alias %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc2_r1_0 = memref.alloc() {address = 0x5100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc2_r1_1 = d2m.operand_alias %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%cast, %cast_0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>)
        outs(%cast_1, %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_ns0, noc = 0>, #d2m.thread<datamovement, @dm_ns1, noc = 1>, #d2m.thread<compute, @cp_ns0>]}
            ins(%view_tc2_r0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc2_r0_0, %cb_tc2_r0_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_ns2, noc = 0>, #d2m.thread<datamovement, @dm_ns3, noc = 1>, #d2m.thread<compute, @cp_ns1>]}
            ins(%view_tc2_r1 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc2_r1_0, %cb_tc2_r1_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }
    %cast_3 = ttir.ttnn_metal_layout_cast %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout2>
    %cast_4 = ttir.ttnn_metal_layout_cast %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout3>
    return %cast_3, %cast_4 : tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>
  }
  func.func private @dm_ns0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_ns1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_ns0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_ns2() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_ns3() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_ns1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----
// TC3: Multi-region, CB 2+3. Region0: 1 in + 1 out (2 CBs). Region1: 2 in + 1 out (3 CBs). in0 shared. Global IOs: in0, in1, out0, out1. CB format buffer_index 0,1 then 0,1,2 in region1 kernels.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram1 = #ttcore.memory_space<dram>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @spatial_multi_cb_2_plus_3
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK: "ttnn.generic"(%arg0, %{{.*}}, %arg1, %{{.*}}) {{.*}}program = #ttnn.program<
  //
  //     Region0 (core (0,0)): 2 CBs; in0 then out0 -> address 0, 1
  //   kernel    | core_ranges | ct_args | common_rt_args | rt_args
  //   ----------|-------------|---------|----------------|--------
  //   dm_2p3_0  | (0,0)       | 0,1     | address<0>     | [*]
  //   dm_2p3_1  | (0,0)       | 0,1     | address<1>     | [*]
  //   cp_2p3_0  | (0,0)       | 0,1     | []             | [*]
  //
  // CHECK-SAME: {{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_2p3_0{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_2p3_1{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}#ttnn.kernel_arg_address_of_tensor<1>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.compute_kernel<symbol_ref = @cp_2p3_0{{.*}}core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>{{.*}}common_rt_args = [], rt_args = [{{[^]]*}}]>,
  //
  //     Region1 (core (1,1)): 3 CBs; shared in0 -> 0, in1 -> 2, out1 -> 3
  //   kernel    | core_ranges | ct_args | common_rt_args | rt_args
  //   ----------|-------------|---------|----------------|--------
  //   dm_2p3_2  | (1,1)       | 0,1,2   | address<0>     | [*]
  //   dm_2p3_3  | (1,1)       | 0,1,2   | address<2>     | [*]
  //   cp_2p3_1  | (1,1)       | 0,1,2   | []             | [*]
  //
  // CHECK-SAME:{{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_2p3_2{{.*}}core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.data_movement_kernel<symbol_ref = @dm_2p3_3{{.*}}core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>{{.*}}#ttnn.kernel_arg_address_of_tensor<2>], rt_args = [{{[^]]*}}]>,
  // CHECK-SAME:{{.*}}#ttnn.compute_kernel<symbol_ref = @cp_2p3_1{{.*}}core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>{{.*}}#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>{{.*}}common_rt_args = [], rt_args = [{{[^]]*}}]>], cbs =
  //
  //     cbs: kernel_cb 2, 3; semaphores
  // CHECK-SAME:{{.*}}kernel_cb_global_buffer_address_of_tensor<1>{{.*}}kernel_cb_global_buffer_address_of_tensor<3>{{.*}}semaphores = []
  func.func @spatial_multi_cb_2_plus_3(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> (tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>) {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %1 = d2m.empty() : tensor<64x64xf32, #ttnn_layout3>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>
    %cast_1 = ttir.ttnn_metal_layout_cast %0 : tensor<64x64xf32, #ttnn_layout2> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %cast_2 = ttir.ttnn_metal_layout_cast %1 : tensor<64x64xf32, #ttnn_layout3> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %view_tc3_r0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc3_r1_0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc3_r1_1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %cb_tc3_r0_0 = memref.alloc() {address = 0x6000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc3_r0_1 = d2m.operand_alias %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc3_r1_0 = memref.alloc() {address = 0x6100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc3_r1_1 = memref.alloc() {address = 0x6200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc3_r1_2 = d2m.operand_alias %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%cast, %cast, %cast_0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>)
        outs(%cast_1, %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_2p3_0, noc = 0>, #d2m.thread<datamovement, @dm_2p3_1, noc = 1>, #d2m.thread<compute, @cp_2p3_0>]}
            ins(%view_tc3_r0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc3_r0_0, %cb_tc3_r0_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_2p3_2, noc = 0>, #d2m.thread<datamovement, @dm_2p3_3, noc = 1>, #d2m.thread<compute, @cp_2p3_1>]}
            ins(%view_tc3_r1_0, %view_tc3_r1_1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc3_r1_0, %cb_tc3_r1_1, %cb_tc3_r1_2 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }
    %cast_3 = ttir.ttnn_metal_layout_cast %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout2>
    %cast_4 = ttir.ttnn_metal_layout_cast %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout3>
    return %cast_3, %cast_4 : tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>
  }
  func.func private @dm_2p3_0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_2p3_1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_2p3_0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_2p3_2() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_2p3_3() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_2p3_1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----
// TC4: Multi-region, 2 shared inputs, 2 outputs. CB 3+3 descriptors; ct_args 0,1,2 per region; address 0,1,2,3. (Full multi-region case last.)
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram1 = #ttcore.memory_space<dram>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @spatial_multi_region
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK: "ttnn.empty"(%{{.*}}) {{.*}}#ttnn.core_range<(0,0), (0,0)>{{.*}}tensor<64x64xf32, #ttnn_layout2>
  // CHECK: "ttnn.empty"(%{{.*}}) {{.*}}#ttnn.core_range<(1,1), (1,1)>{{.*}}tensor<64x64xf32, #ttnn_layout3>
  // CHECK: "ttnn.generic"(%arg0, %arg1, %{{.*}}, %{{.*}}) {{.*}}program = #ttnn.program<
  //
  // Region0 / region1: unified tensor addresses; CB ct_args stay 0,1,2 per region (full kernel snippets; no greedy FileCheck gaps).
  // CHECK: #ttnn.data_movement_kernel<symbol_ref = @dm_k0, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = []>
  // CHECK: #ttnn.data_movement_kernel<symbol_ref = @dm_k1, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, processor = riscv0, noc_index = noc1, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<1>], rt_args = []>
  // CHECK: #ttnn.data_movement_kernel<symbol_ref = @dm_k2, core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = []>
  // CHECK: #ttnn.data_movement_kernel<symbol_ref = @dm_k3, core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>, processor = riscv0, noc_index = noc1, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<1>], rt_args = []>
  // CHECK: kernel_cb_global_buffer_address_of_tensor<2>{{.*}}kernel_cb_global_buffer_address_of_tensor<3>{{.*}}semaphores = []
  func.func @spatial_multi_region(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> (tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>) {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %1 = d2m.empty() : tensor<64x64xf32, #ttnn_layout3>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>
    %cast_1 = ttir.ttnn_metal_layout_cast %0 : tensor<64x64xf32, #ttnn_layout2> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %cast_2 = ttir.ttnn_metal_layout_cast %1 : tensor<64x64xf32, #ttnn_layout3> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %view_tc4_r0_0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc4_r0_1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc4_r1_0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc4_r1_1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %cb_tc4_r0_0 = memref.alloc() {address = 0x7000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc4_r0_1 = memref.alloc() {address = 0x7100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc4_r0_2 = d2m.operand_alias %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc4_r1_0 = memref.alloc() {address = 0x7200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc4_r1_1 = memref.alloc() {address = 0x7300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc4_r1_2 = d2m.operand_alias %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%cast, %cast_0, %cast, %cast_0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>)
        outs(%cast_1, %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_k0, noc = 0>, #d2m.thread<datamovement, @dm_k1, noc = 1>, #d2m.thread<compute, @cp_k0>]}
            ins(%view_tc4_r0_0, %view_tc4_r0_1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc4_r0_0, %cb_tc4_r0_1, %cb_tc4_r0_2 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_k2, noc = 0>, #d2m.thread<datamovement, @dm_k3, noc = 1>, #d2m.thread<compute, @cp_k1>]}
            ins(%view_tc4_r1_0, %view_tc4_r1_1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc4_r1_0, %cb_tc4_r1_1, %cb_tc4_r1_2 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }
    %cast_3 = ttir.ttnn_metal_layout_cast %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout2>
    %cast_4 = ttir.ttnn_metal_layout_cast %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout3>
    return %cast_3, %cast_4 : tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>
  }
  func.func private @dm_k0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_k1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_k0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_k2() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_k3() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_k1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>, <arg_type = cb_port, operand_index = 5>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----
// TC5: Multi-region with general semaphores. Per-region semaphore_at indices stay 0,1;
// merged program lists four semaphore descriptors (id 0,1 on (0,0) then id 0,1 on (1,1)).
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram1 = #ttcore.memory_space<dram>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @spatial_multi_region_semaphore
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // TC5: Each region keeps semaphore_at<0,1>; merged semaphores list id 0,1 per core range (duplicate ids).
  // CHECK: #ttnn.data_movement_kernel<symbol_ref = @dm_r0_sem, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_semaphore_at<0>, #ttnn.kernel_arg_semaphore_at<1>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = []>
  // CHECK: #ttnn.data_movement_kernel<symbol_ref = @dm_r1_sem, core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_semaphore_at<0>, #ttnn.kernel_arg_semaphore_at<1>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<2>], rt_args = []>
  // CHECK: semaphores = [<id = 0, core_type = worker, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, initial_value = 0>
  // CHECK: <id = 1, core_type = worker, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>, initial_value = 0>
  // CHECK: <id = 0, core_type = worker, core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>, initial_value = 0>
  // CHECK: <id = 1, core_type = worker, core_ranges = <[#ttnn.core_range<(1,1), (1,1)>]>, initial_value = 0>
  func.func @spatial_multi_region_semaphore(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> (tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>) {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %1 = d2m.empty() : tensor<64x64xf32, #ttnn_layout3>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>
    %cast_1 = ttir.ttnn_metal_layout_cast %0 : tensor<64x64xf32, #ttnn_layout2> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %cast_2 = ttir.ttnn_metal_layout_cast %1 : tensor<64x64xf32, #ttnn_layout3> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %view_tc5_r0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc5_r1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %cb_tc5_r0_0 = memref.alloc() {address = 0x8000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc5_r0_1 = memref.alloc() {address = 0x8100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc5_r1_0 = memref.alloc() {address = 0x8200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc5_r1_1 = memref.alloc() {address = 0x8300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%cast, %cast_0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>)
        outs(%cast_1, %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r0_sem, noc = 0>, #d2m.thread<datamovement, @dm_r0_nosem, noc = 1>, #d2m.thread<compute, @cp_r0>]}
            ins(%view_tc5_r0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc5_r0_0, %cb_tc5_r0_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_r1_sem, noc = 0>, #d2m.thread<datamovement, @dm_r1_nosem, noc = 1>, #d2m.thread<compute, @cp_r1>]}
            ins(%view_tc5_r1 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%cb_tc5_r1_0, %cb_tc5_r1_1 : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }
    %cast_3 = ttir.ttnn_metal_layout_cast %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout2>
    %cast_4 = ttir.ttnn_metal_layout_cast %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout3>
    return %cast_3, %cast_4 : tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>
  }
  func.func private @dm_r0_sem() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>, <arg_type = local_semaphore, operand_index = 0>, <arg_type = local_semaphore, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_r0_nosem() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_r1_sem() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>, <arg_type = local_semaphore, operand_index = 0>, <arg_type = local_semaphore, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_r1_nosem() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_r1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}

// -----
// TC6: Two regions, one global semaphore additional arg per region.
// additional args append sem0 then sem1 -> kernel_arg_global_semaphore<4> and <5>.
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram1 = #ttcore.memory_space<dram>
#l1_1 = #ttcore.memory_space<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, exactGrid = true>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1, d2) -> (d1 - 1, d2 - 1)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @spatial_two_global_semaphores
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // Merged operand list: four IO tensors then two !ttnn.global_semaphore; per-region local operand_index 2 maps to flat 4 then 5.
  // CHECK: "ttnn.generic"
  // CHECK-SAME: symbol_ref = @dm_g0_0
  // CHECK-SAME: kernel_arg_global_semaphore<4>
  // CHECK-SAME: symbol_ref = @dm_g0_1
  // CHECK-SAME: kernel_arg_global_semaphore<4>
  // CHECK-SAME: symbol_ref = @dm_g1_0
  // CHECK-SAME: kernel_arg_global_semaphore<5>
  // CHECK-SAME: symbol_ref = @dm_g1_1
  // CHECK-SAME: kernel_arg_global_semaphore<5>
  func.func @spatial_two_global_semaphores(%arg0: tensor<64x128xf32, #ttnn_layout>, %arg1: tensor<128x64xf32, #ttnn_layout1>) -> (tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>) {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout2>
    %1 = d2m.empty() : tensor<64x64xf32, #ttnn_layout3>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn_layout> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg1 : tensor<128x64xf32, #ttnn_layout1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>
    %cast_1 = ttir.ttnn_metal_layout_cast %0 : tensor<64x64xf32, #ttnn_layout2> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %cast_2 = ttir.ttnn_metal_layout_cast %1 : tensor<64x64xf32, #ttnn_layout3> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>
    %alloc_sem0 = memref.alloc() {address = 200000 : i64, alignment = 16 : i64} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1_1>
    %alloc_sem1 = memref.alloc() {address = 200016 : i64, alignment = 16 : i64} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1_1>
    %sem0 = d2m.create_global_semaphore(%alloc_sem0) {value = 0 : ui32} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1_1> -> !d2m.global_semaphore
    %sem1 = d2m.create_global_semaphore(%alloc_sem1) {value = 0 : ui32} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1_1> -> !d2m.global_semaphore
    %view_tc6_r0 = d2m.view_layout %cast remapping = #map3 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %view_tc6_r1 = d2m.view_layout %cast_0 remapping = #map3 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1> -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>
    %cb_tc6_r0_0 = memref.alloc() {address = 0x9000 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc6_r0_1 = memref.alloc() {address = 0x9100 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc6_r1_0 = memref.alloc() {address = 0x9200 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    %cb_tc6_r1_1 = memref.alloc() {address = 0x9300 : i64, alignment = 16 : i64} : memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%cast, %cast_0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram1>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #dram1>)
        outs(%cast_1, %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>) {
      ^region_0:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_g0_0, noc = 0>, #d2m.thread<datamovement, @dm_g0_1, noc = 1>, #d2m.thread<compute, @cp_g0>]}
            ins(%view_tc6_r0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%sem0, %cb_tc6_r0_0, %cb_tc6_r0_1 : !d2m.global_semaphore, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }, {
      ^region_1:
        d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @dm_g1_0, noc = 0>, #d2m.thread<datamovement, @dm_g1_1, noc = 1>, #d2m.thread<compute, @cp_g1>]}
            ins(%view_tc6_r1 : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram1>)
            outs(%cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1>)
            additionalArgs(%sem1, %cb_tc6_r1_0, %cb_tc6_r1_1 : !d2m.global_semaphore, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>, memref<2x2x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 1>, #l1_1>)
      }
    d2m.reset_global_semaphore(%sem0) {value = 0 : ui32} : !d2m.global_semaphore
    d2m.reset_global_semaphore(%sem1) {value = 0 : ui32} : !d2m.global_semaphore
    memref.dealloc %alloc_sem1 : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1_1>
    memref.dealloc %alloc_sem0 : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1_1>
    %cast_3 = ttir.ttnn_metal_layout_cast %cast_1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout2>
    %cast_4 = ttir.ttnn_metal_layout_cast %cast_2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_1> -> tensor<64x64xf32, #ttnn_layout3>
    return %cast_3, %cast_4 : tensor<64x64xf32, #ttnn_layout2>, tensor<64x64xf32, #ttnn_layout3>
  }
  func.func private @dm_g0_0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = global_semaphore, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_g0_1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_g0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @dm_g1_0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>, <arg_type = global_semaphore, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @dm_g1_1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>] ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @cp_g1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 4>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}
