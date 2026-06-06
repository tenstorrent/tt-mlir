// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

module {
  // CHECK-LABEL: @test_integer_prod
  // CHECK-NOT: "ttnn.prod"
  // CHECK: "ttnn.slice_static"
  // CHECK: "ttnn.slice_static"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.slice_static"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.reshape"
  func.func public @test_integer_prod(%arg0: tensor<1x3xsi32, #ttnn_layout>) -> tensor<1xsi32, #ttnn_layout1> {
    %0 = "ttnn.prod"(%arg0) <{dim_arg = 1 : i64, keep_dim = false}> : (tensor<1x3xsi32, #ttnn_layout>) -> tensor<1xsi32, #ttnn_layout1>
    return %0 : tensor<1xsi32, #ttnn_layout1>
  }
}
