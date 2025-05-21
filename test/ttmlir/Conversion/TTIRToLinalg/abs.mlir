// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s



module attributes {} {
  func.func @hoisted(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "ttir.empty"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "ttir.abs"(%arg0, %out) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}


module attributes {} {
  func.func @hoisted(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %1 = "tosa.abs"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}


module attributes {} {
  func.func @hoisted(%arg0: tensor<64x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = "tensor.empty"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "linagl.generic"(%arg0, %0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}


module attributes {} {
  func.func @hoisted(%arg0: tensor<64x128xf32>, %out: tensor<64x128xf32>) {
    %0 = "tensor.empty"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = "linagl.generic"(%arg0, %out) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1
  }
}


module attributes {} {

  func.func @hoisted(%arg0: tensor<64x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32>;

  func.func @test_abs(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: {{%[0-9]+}} = tensor.empty() : {{tensor<.*>}}
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: {{%[0-9]+}} = linalg.abs ins(%arg0 : {{tensor<.*>}}) outs({{%[0-9]+}} : {{tensor<.*>}}) -> {{tensor<.*>}}
    %1 = "ttir.abs"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: return %1 : {{tensor<.*>}}
    return %1 : tensor<64x128xf32>
  }

}
