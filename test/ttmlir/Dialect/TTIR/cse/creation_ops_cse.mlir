// RUN: ttmlir-opt -canonicalize -cse -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that identical zeros ops are CSE'd into a single op.
// CHECK-LABEL: @zeros_cse
func.func @zeros_cse(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  // CHECK: "ttir.zeros"
  %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
  // CHECK-NOT: "ttir.zeros"
  %1 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
  return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// Test that identical ones ops are CSE'd into a single op.
// CHECK-LABEL: @ones_cse
func.func @ones_cse(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  // CHECK: "ttir.ones"
  %0 = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
  // CHECK-NOT: "ttir.ones"
  %1 = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
  return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// Test that identical full ops are CSE'd into a single op.
// CHECK-LABEL: @full_cse
func.func @full_cse() -> (tensor<64x64xbf16>, tensor<64x64xbf16>) {
  // CHECK: "ttir.full"
  %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 3.0 : f32}> : () -> tensor<64x64xbf16>
  // CHECK-NOT: "ttir.full"
  %1 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 3.0 : f32}> : () -> tensor<64x64xbf16>
  return %0, %1 : tensor<64x64xbf16>, tensor<64x64xbf16>
}

// Test that identical arange ops are CSE'd into a single op.
// CHECK-LABEL: @arange_cse
func.func @arange_cse() -> (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) {
  // CHECK: "ttir.arange"
  %0 = "ttir.arange"() <{start = 0 : si64, end = 128 : si64, step = 1 : si64, arange_dimension = 3 : i64}> : () -> tensor<1x1x32x128xbf16>
  // CHECK-NOT: "ttir.arange"
  %1 = "ttir.arange"() <{start = 0 : si64, end = 128 : si64, step = 1 : si64, arange_dimension = 3 : i64}> : () -> tensor<1x1x32x128xbf16>
  return %0, %1 : tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>
}

// Test that identical empty ops are NOT CSE'd — each is a distinct allocation.
// CHECK-LABEL: @empty_no_cse
func.func @empty_no_cse() -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  // CHECK: ttir.empty
  %0 = ttir.empty() : tensor<64x64xf32>
  // CHECK: ttir.empty
  %1 = ttir.empty() : tensor<64x64xf32>
  return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// Test that empty ops used as DPS operands for to_layout are not merged.
// CHECK-LABEL: @empty_to_layout_no_cse
func.func @empty_to_layout_no_cse(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  // CHECK: ttir.empty
  %0 = ttir.empty() : tensor<32x32xf32>
  // CHECK: ttir.to_layout
  %1 = ttir.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
  // CHECK: ttir.empty
  %2 = ttir.empty() : tensor<32x32xf32>
  // CHECK: ttir.to_layout
  %3 = ttir.to_layout %arg1, %2 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
  return %1, %3 : tensor<32x32xf32>, tensor<32x32xf32>
}

// Test that unused empty ops are removed by DCE despite having an Allocate
// memory effect. The Allocate effect prevents CSE from merging distinct empty
// ops, but DCE can still remove unused ones.
// CHECK-LABEL: @empty_unused_dce
func.func @empty_unused_dce(%arg0: tensor<32x32xbf16>) -> tensor<32x32xf32> {
  // CHECK: ttir.empty
  %0 = ttir.empty() : tensor<32x32xf32>
  // CHECK: ttir.to_layout
  %1 = ttir.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<32x32xf32> -> tensor<32x32xf32>
  // CHECK-NOT: ttir.empty
  %unused = ttir.empty() : tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
