// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_input = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 30 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_result = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

func.func @test_repeat_int32(%arg0: tensor<1x1x32xsi32, #ttnn_layout_input>) -> tensor<1x16x32xsi32, #ttnn_layout_result> {
  // CHECK: "ttnn.zeros"
  // CHECK: "ttnn.add"
  // CHECK-NOT: "ttnn.repeat"
  %0 = "ttnn.repeat"(%arg0) {repeat_dims = #ttnn.shape<1x16x1>} : (tensor<1x1x32xsi32, #ttnn_layout_input>) -> tensor<1x16x32xsi32, #ttnn_layout_result>
  return %0 : tensor<1x16x32xsi32, #ttnn_layout_result>
}
