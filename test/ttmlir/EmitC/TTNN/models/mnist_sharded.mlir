// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" %s -o %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// Temporary workaround for running optimizer tests in CI is to run optimizer locally and run the rest of the pipeline on obtained IR:
// https://github.com/tenstorrent/tt-mlir/issues/3717

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x25x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<25x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

func.func @mnist_fwd(%arg0: tensor<1x784xf32, #ttnn_layout>, %arg1: tensor<1x10xf32, #ttnn_layout1>, %arg2: tensor<256x10xf32, #ttnn_layout2>, %arg3: tensor<1x256xf32, #ttnn_layout3>, %arg4: tensor<784x256xf32, #ttnn_layout4>) -> tensor<1x10xf32, #ttnn_layout5> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.matmul"(%arg0, %arg4) <{transpose_a = false, transpose_b = false}> : (tensor<1x784xf32, #ttnn_layout>, tensor<784x256xf32, #ttnn_layout4>) -> tensor<1x256xf32, #ttnn_layout5>
  "ttnn.deallocate"(%arg4) <{force = false}> : (tensor<784x256xf32, #ttnn_layout4>) -> ()
  "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x784xf32, #ttnn_layout>) -> ()
  %2 = "ttnn.add"(%1, %arg3) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x256xf32, #ttnn_layout5>, tensor<1x256xf32, #ttnn_layout3>) -> tensor<1x256xf32, #ttnn_layout5>
  "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x256xf32, #ttnn_layout5>) -> ()
  "ttnn.deallocate"(%arg3) <{force = false}> : (tensor<1x256xf32, #ttnn_layout3>) -> ()
  %3 = "ttnn.to_memory_config"(%2) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (7,0)>]>, <32x32>, <row_major>, <physical>>>}> : (tensor<1x256xf32, #ttnn_layout5>) -> tensor<1x256xf32, #ttnn_layout6>
  "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x256xf32, #ttnn_layout5>) -> ()
  %4 = "ttnn.relu"(%3) : (tensor<1x256xf32, #ttnn_layout6>) -> tensor<1x256xf32, #ttnn_layout6>
  "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x256xf32, #ttnn_layout6>) -> ()
  %5 = "ttnn.to_memory_config"(%4) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x256xf32, #ttnn_layout6>) -> tensor<1x256xf32, #ttnn_layout1>
  "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x256xf32, #ttnn_layout6>) -> ()
  %6 = "ttnn.matmul"(%5, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x256xf32, #ttnn_layout1>, tensor<256x10xf32, #ttnn_layout2>) -> tensor<1x10xf32, #ttnn_layout5>
  "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x256xf32, #ttnn_layout1>) -> ()
  "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<256x10xf32, #ttnn_layout2>) -> ()
  %7 = "ttnn.add"(%6, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x10xf32, #ttnn_layout5>, tensor<1x10xf32, #ttnn_layout1>) -> tensor<1x10xf32, #ttnn_layout5>
  "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x10xf32, #ttnn_layout5>) -> ()
  "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x10xf32, #ttnn_layout1>) -> ()
  %8 = "ttnn.softmax"(%7) <{dimension = 1 : si32}> : (tensor<1x10xf32, #ttnn_layout5>) -> tensor<1x10xf32, #ttnn_layout5>
  "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x10xf32, #ttnn_layout5>) -> ()
  return %8 : tensor<1x10xf32, #ttnn_layout5>
}
