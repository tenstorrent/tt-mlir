// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module @Model attributes {} {
  func.func @forward(%arg0: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"}, %arg1: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "l.weight"}, %arg2: tensor<10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "l.bias"}, %arg3: tensor<10x10xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "extra_tensor"}) -> (tensor<10x10xf32> {ttir.name = "Model.output_add_2"}) {
    %0 = ttir.empty() : tensor<10x10xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %arg2, %0) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %2 = ttir.empty() : tensor<10x10xf32>
    %3 = "ttir.add"(%1, %arg3, %2) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    return %3 : tensor<10x10xf32>
  }
}
