// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false enable-const-eval=true" -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// CHECK: func.func @integer_input_rm_prop
// CHECK-NOT: "ttnn.to_layout"
// CHECK: = "ttnn.add"(%arg0, %arg1)

func.func @integer_input_rm_prop(%arg0: tensor<32x32xsi32> {ttcore.argument_type = #ttcore.argument_type<input>},
                                 %arg1: tensor<32x32xsi32> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xsi32> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xsi32>, tensor<32x32xsi32>) -> tensor<32x32xsi32>
    return %0 : tensor<32x32xsi32>
}
