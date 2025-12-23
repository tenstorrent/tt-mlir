// RUN: ttmlir-opt --ttir-to-ttir-decomposition="config=cpu-fallback" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func public @test_reduce_or_4to3dim(%arg0: tensor<128x10x32x4xbf16>, %arg1: tensor<1xbf16>) -> tensor<128x10x32xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_or_4to3dim
    // CHECK-NOT: "ttir.reduce_or"

    // Sum
    // CHECK: %[[SUM:.*]] = "ttir.sum"(%{{.*}}) <{dim_arg = [3 : i32], keep_dim = false}>
    // CHECK-SAME: : (tensor<128x10x32x4xbf16>) -> tensor<128x10x32xbf16>

    // Zero constant
    // CHECK: %[[C0:.*]] = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<128x10x32xbf16>}>

    // Ne
    // CHECK: %[[NE:.*]] = "ttir.ne"(%[[SUM]], %[[C0]])
    // CHECK-SAME: : (tensor<128x10x32xbf16>, tensor<128x10x32xbf16>) -> tensor<128x10x32xbf16>

    // Typecast
    // CHECK: %[[CAST:.*]] = "ttir.typecast"(%[[NE]])

    // CHECK: return %[[CAST]] : tensor<128x10x32xbf16>
    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<128x10x32x4xbf16>) -> tensor<128x10x32xbf16>
    return %0 : tensor<128x10x32xbf16>
  }

  func.func public @test_reduce_or_3to2dim(%arg0: tensor<128x10x4xbf16>, %arg1: tensor<1xbf16>) -> tensor<128x4xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_or_3to2dim
    // CHECK-NOT: "ttir.reduce_or"

    // Sum
    // CHECK: %[[SUM:.*]] = "ttir.sum"(%{{.*}}) <{dim_arg = [1 : i32], keep_dim = false}>
    // CHECK-SAME: : (tensor<128x10x4xbf16>) -> tensor<128x4xbf16>

    // Zero constant
    // CHECK: %[[C0:.*]] = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<128x4xbf16>}>

    // Ne
    // CHECK: %[[NE:.*]] = "ttir.ne"(%[[SUM]], %[[C0]])
    // CHECK-SAME: : (tensor<128x4xbf16>, tensor<128x4xbf16>) -> tensor<128x4xbf16>

    // Typecast
    // CHECK: %[[CAST:.*]] = "ttir.typecast"(%[[NE]])

    // CHECK: return %[[CAST]] : tensor<128x4xbf16>

    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<128x10x4xbf16>) -> tensor<128x4xbf16>
    return %0 : tensor<128x4xbf16>
  }

  func.func public @test_reduce_or_2to1dim(%arg0: tensor<128x10xbf16>, %arg1: tensor<1xbf16>) -> tensor<10xbf16> {
    // CHECK-LABEL: func.func public @test_reduce_or_2to1dim
    // CHECK-NOT: "ttir.reduce_or"

    // Sum
    // CHECK: %[[SUM:.*]] = "ttir.sum"(%{{.*}}) <{dim_arg = [0 : i32], keep_dim = false}>
    // CHECK-SAME: : (tensor<128x10xbf16>) -> tensor<10xbf16>

    // Zero constant
    // CHECK: %[[C0:.*]] = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<10xbf16>}>

    // Ne
    // CHECK: %[[NE:.*]] = "ttir.ne"(%[[SUM]], %[[C0]])
    // CHECK-SAME: : (tensor<10xbf16>, tensor<10xbf16>) -> tensor<10xbf16>

    // Typecast
    // CHECK: %[[CAST:.*]] = "ttir.typecast"(%[[NE]])

    // CHECK: return %[[CAST]] : tensor<10xbf16>

    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<128x10xbf16>) -> tensor<10xbf16>
    return %0 : tensor<10xbf16>
  }
}
