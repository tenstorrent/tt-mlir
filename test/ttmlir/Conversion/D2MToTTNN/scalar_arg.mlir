// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --convert-d2m-to-ttnn -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test that a scalar (integer/index/float) operand in a d2m.generic's
// additionalArgs is converted into an additional arg on the ttnn.generic and
// that the kernel's scalar arg_spec is remapped to the scalar's position in
// the ttnn.generic's additionalArgs list (CBs/semaphores do not appear there).
// See "Handle scalar args in D2M to TTNN".

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#l1_mem = #ttcore.memory_space<l1>

#core_range = #ttnn.core_range<(0,0), (0,0)>
#core_ranges = #ttnn.core_range_set<[#core_range]>

#dram_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
  >
#l1_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = #core_ranges
  >

module {
  // CHECK-LABEL: func.func @test_scalar_arg
  // The scalar function argument is threaded through to the ttnn.generic.
  // CHECK-SAME: %arg1: i32
  func.func @test_scalar_arg(%arg0: tensor<32x32xf32, #dram_layout>, %scalar: i32) -> tensor<32x32xf32, #dram_layout> {
    %device = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %ttnn_input_l1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
    %ttnn_output_l1 = d2m.empty() : tensor<32x32xf32, #l1_layout>
    %metal_input_l1 = ttir.ttnn_metal_layout_cast %ttnn_input_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %metal_output_l1 = ttir.ttnn_metal_layout_cast %ttnn_output_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %view_input = d2m.view_layout %metal_input_l1 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %view_output = d2m.view_layout %metal_output_l1 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %cb_0 = d2m.operand_alias %view_input : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1_mem>
    %cb_1 = d2m.operand_alias %view_output : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1_mem>

    // The scalar (operand_index 4 in the d2m.generic, after 2 io tensors + 2
    // CBs) becomes a ttnn.generic additional arg. The io tensor count is 2 and
    // CBs/semaphores are not added to the ttnn.generic's additional args, so the
    // scalar lands at index 2 -- kernel_arg_scalar must be remapped to <2>, not
    // its raw d2m operand index <4>.
    // CHECK: "ttnn.generic"(%{{.*}}, %{{.*}}, %arg1)
    // CHECK-SAME: operandSegmentSizes = array<i32: 2, 1>
    // CHECK-SAME: ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<1>]
    // CHECK-SAME: common_rt_args = [#ttnn.kernel_arg_scalar<2>]
    // CHECK-SAME: (tensor<32x32xf32, {{.*}}>, tensor<32x32xf32, {{.*}}>, i32) -> ()
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute, @compute_kernel0>]}
        ins(%view_input : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
        outs(%view_output : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
    additionalArgs(%cb_0, %cb_1, %scalar : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1_mem>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 1>, #l1_mem>, i32)

    %output_l1 = ttir.ttnn_metal_layout_cast %metal_output_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> tensor<32x32xf32, #l1_layout>
    %output_dram = "ttnn.to_memory_config"(%output_l1) : (tensor<32x32xf32, #l1_layout>) -> tensor<32x32xf32, #dram_layout>
    return %output_dram : tensor<32x32xf32, #dram_layout>
  }
  func.func private @compute_kernel0() attributes {
    ttkernel.arg_spec = #ttkernel.arg_spec<
      rt_args = [
        <arg_type = scalar, operand_index = 4>
      ]
      ct_args = [
        <arg_type = cb_port, operand_index = 2>,
        <arg_type = cb_port, operand_index = 3>
      ]
    >,
    ttkernel.thread = #ttkernel.thread<compute>
  } {
    return
  }
}
