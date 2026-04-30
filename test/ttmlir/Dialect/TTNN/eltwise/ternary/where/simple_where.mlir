// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_where {
  func.func public @test_where(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
    %0 = ttir.empty() : tensor<13x37xf32>
    %1 = "ttir.eq"(%arg0, %arg1) : (tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    %3 = "ttir.where"(%1, %arg0, %arg1) : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    // CHECK: "ttnn.eq"
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: -> tensor<13x37xf32
    // CHECK: "ttnn.where"
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: tensor<13x37xf32
    // CHECK-SAME: -> tensor<13x37xf32
     return %3 : tensor<13x37xf32>
  }

  func.func public @where_predicate_different_than_input(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xbf16>, %arg2: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
    %0 = ttir.empty() : tensor<13x37xbf16>
    // Predicate (f32) differs from float input (bf16) — typecast predicate.
    // CHECK: "ttnn.typecast"{{.*}}{dtype = #ttcore.supportedDataTypes<bf16>}
    // CHECK: "ttnn.where"{{.*}}
    %1 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<13x37xf32>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
    return %1 : tensor<13x37xbf16>
  }

  func.func public @where_predicate_integer(%arg0: tensor<13x37xsi32>, %arg1: tensor<13x37xbf16>, %arg2: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
    %0 = ttir.empty() : tensor<13x37xbf16>
    // Predicate (si32) differs from float input (bf16) — typecast predicate.
    // CHECK: "ttnn.typecast"{{.*}}{dtype = #ttcore.supportedDataTypes<bf16>}
    // CHECK: "ttnn.where"{{.*}}
    %1 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<13x37xsi32>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
    return %1 : tensor<13x37xbf16>
  }

  func.func public @where_all_operands_integer(%arg0: tensor<13x37xsi32>, %arg1: tensor<13x37xsi32>, %arg2: tensor<13x37xsi32>) -> tensor<13x37xsi32> {
    // si32 is natively supported — no typecast needed (si32→si32 is no-op).
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.where"{{.*}}
    %1 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<13x37xsi32>, tensor<13x37xsi32>, tensor<13x37xsi32>) -> tensor<13x37xsi32>
    return %1 : tensor<13x37xsi32>
  }

  func.func public @where_all_i8(%arg0: tensor<13x37xi8>, %arg1: tensor<13x37xi8>, %arg2: tensor<13x37xi8>) -> tensor<13x37xi8> {
    // i8 not natively supported — cast to si32, result cast back.
    // CHECK: "ttnn.typecast"{{.*}}{dtype = #ttcore.supportedDataTypes<si32>}
    // CHECK: "ttnn.where"
    // CHECK: "ttnn.typecast"
    %1 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<13x37xi8>, tensor<13x37xi8>, tensor<13x37xi8>) -> tensor<13x37xi8>
    return %1 : tensor<13x37xi8>
  }
}
