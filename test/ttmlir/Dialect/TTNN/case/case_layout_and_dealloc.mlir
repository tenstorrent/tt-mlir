// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @three_branch

// The index is read back to host to pick a branch, so it has to end up as an
// si32 tensor in system memory. It stays signed: a negative index has to keep
// reading as out of range, which selects the last branch.
// CHECK: ttnn.from_device
// CHECK-SAME: -> tensor<si32
// CHECK: ttnn.case index(%{{[0-9]+}} : tensor<si32

// Branch block arguments are the captures, owned by the caller, so a branch must
// never deallocate them. These CHECK-NOTs are bounded by the surrounding
// CHECKs, i.e. they only cover the inside of the branches.
// CHECK-NOT: "ttnn.deallocate"(%arg
// CHECK: ttnn.yield %{{[0-9]+}} : tensor<32x32xf32
// CHECK: ^bb0(
// CHECK-NOT: "ttnn.deallocate"(%arg
// CHECK: ttnn.yield %{{[0-9]+}} : tensor<32x32xf32
// CHECK: ^bb0(
// CHECK-NOT: "ttnn.deallocate"(%arg

// Every branch yields the op's result type exactly; otherwise the consumer would
// read one descriptor for values that disagree. The op verifier enforces it, so
// reaching the end of the pipeline is the proof.
// CHECK: } -> (tensor<32x32xf32

func.func @three_branch(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %two = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0, %two : tensor<32x32xf32>, tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.multiply"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.subtract"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// Conv3d is the one op whose result TTNNLayout leaves row-major, while the case
// returns its results tiled, so its branch has to relay out what it yields.
// CHECK-LABEL: func.func @yield_disagrees_on_layout
// CHECK: ttnn.case
// CHECK: %[[ADD:[0-9]+]] = "ttnn.add"
// CHECK-NOT: ttnn.to_layout
// CHECK: ttnn.yield %[[ADD]]
// CHECK: %[[CONV:[0-9]+]] = "ttnn.conv3d"
// CHECK: %[[RELAID:[0-9]+]] = "ttnn.to_layout"(%[[CONV]])
// CHECK: ttnn.yield %[[RELAID]]
func.func @yield_disagrees_on_layout(%input: tensor<1x8x28x28x4xf32>, %weight: tensor<16x4x3x3x3xf32>, %other: tensor<1x6x26x26x16xf32>, %index: tensor<i32>) -> tensor<1x6x26x26x16xf32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%input, %weight, %other : tensor<1x8x28x28x4xf32>, tensor<16x4x3x3x3xf32>, tensor<1x6x26x26x16xf32>)
  branches {
  ^bb0(%i: tensor<1x8x28x28x4xf32>, %w: tensor<16x4x3x3x3xf32>, %o: tensor<1x6x26x26x16xf32>):
    %0 = "ttir.add"(%o, %o) : (tensor<1x6x26x26x16xf32>, tensor<1x6x26x26x16xf32>) -> tensor<1x6x26x26x16xf32>
    ttir.yield %0 : tensor<1x6x26x26x16xf32>
  }, {
  ^bb0(%i: tensor<1x8x28x28x4xf32>, %w: tensor<16x4x3x3x3xf32>, %o: tensor<1x6x26x26x16xf32>):
    %w2 = "ttir.multiply"(%w, %w) : (tensor<16x4x3x3x3xf32>, tensor<16x4x3x3x3xf32>) -> tensor<16x4x3x3x3xf32>
    %0 = "ttir.conv3d"(%i, %w2) <{stride = array<i32: 1, 1, 1>, padding = array<i32: 0, 0, 0>, groups = 1 : i32, padding_mode = "zeros"}> : (tensor<1x8x28x28x4xf32>, tensor<16x4x3x3x3xf32>) -> tensor<1x6x26x26x16xf32>
    ttir.yield %0 : tensor<1x6x26x26x16xf32>
  } -> (tensor<1x6x26x26x16xf32>)
  return %r : tensor<1x6x26x26x16xf32>
}

// A row-major capture has to be relaid out to the branch block arguments,
// which are tiled, before it enters the case.
// CHECK-LABEL: func.func @capture_disagrees_on_layout
// CHECK: %[[CONV:[0-9]+]] = "ttnn.conv3d"
// CHECK: %[[CAPTURE:[0-9]+]] = "ttnn.to_layout"(%[[CONV]])
// CHECK: ttnn.case index(%{{[0-9]+}} : {{.*}}) captures(%[[CAPTURE]],
func.func @capture_disagrees_on_layout(%input: tensor<1x8x28x28x4xf32>, %weight: tensor<16x4x3x3x3xf32>, %other: tensor<1x6x26x26x16xf32>, %index: tensor<i32>) -> tensor<1x6x26x26x16xf32> {
  %conv = "ttir.conv3d"(%input, %weight) <{stride = array<i32: 1, 1, 1>, padding = array<i32: 0, 0, 0>, groups = 1 : i32, padding_mode = "zeros"}> : (tensor<1x8x28x28x4xf32>, tensor<16x4x3x3x3xf32>) -> tensor<1x6x26x26x16xf32>
  %r = ttir.case index(%index : tensor<i32>) captures(%conv, %other : tensor<1x6x26x26x16xf32>, tensor<1x6x26x26x16xf32>)
  branches {
  ^bb0(%c: tensor<1x6x26x26x16xf32>, %o: tensor<1x6x26x26x16xf32>):
    %0 = "ttir.add"(%o, %o) : (tensor<1x6x26x26x16xf32>, tensor<1x6x26x26x16xf32>) -> tensor<1x6x26x26x16xf32>
    ttir.yield %0 : tensor<1x6x26x26x16xf32>
  }, {
  ^bb0(%c: tensor<1x6x26x26x16xf32>, %o: tensor<1x6x26x26x16xf32>):
    ttir.yield %c : tensor<1x6x26x26x16xf32>
  } -> (tensor<1x6x26x26x16xf32>)
  return %r : tensor<1x6x26x26x16xf32>
}
