// RUN: ttmlir-opt --ttcore-register-device --canonicalize --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_arange_output = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_reshape_1_output = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_reshape_2_output = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x32x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>
// CHECK: #[[TTNN_LAYOUT_ARANGE_OUTPUT:.*]] = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1024xsi32, #dram>, <interleaved>>
module {
    func.func @arange_workaround_test() -> (tensor<1x1024x1xsi32, #ttnn_layout_reshape_1_output>, tensor<1x1x1024xsi32, #ttnn_layout_reshape_2_output>) {
      %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
      %1 = "ttnn.arange"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, end = 1024 : i64, memory_config = #ttnn.memory_config<<dram>, <interleaved>>, start = 0 : i64, step = 1 : i64}> : (!ttnn.device) -> tensor<1024xsi32, #ttnn_layout_arange_output>
      // Verify that the output of the arange operation is row-major.
      // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
      // CHECK-NEXT: %[[ARANGE_OP:.*]] = "ttnn.arange"(%[[GET_DEVICE_OP]])
      // CHECK-SAME: -> tensor<1024xsi32, #[[TTNN_LAYOUT_ARANGE_OUTPUT]]>
      // Verify that after arange op, there is a to layout op to convert the layout to tile.
      // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[ARANGE_OP]])
      // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
      // CHECK-SAME: layout = #ttnn.layout<tile>
      // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
      %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1024 : i32, 1 : i32]}> : (tensor<1024xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1024x1xsi32, #ttnn_layout_reshape_1_output>
      %3 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1x1024xsi32, #ttnn_layout_reshape_2_output>
      return %2, %3 : tensor<1x1024x1xsi32, #ttnn_layout_reshape_1_output>, tensor<1x1x1024xsi32, #ttnn_layout_reshape_2_output>
    }
}
