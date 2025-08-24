// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
    func.func public @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
        %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
        // CHECK: = ttir.empty
        // CHECK: = "ttir.broadcast"
        // CHECK: = ttir.empty
        // CHECK: = "ttir.reshape"
        %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
        // CHECK: = ttir.empty
        // CHECK: = "ttir.broadcast"
        %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
        // CHECK: = ttir.empty
        // CHECK: = "ttir.add"
        return %2 : tensor<1x128xf32>
    }
}
