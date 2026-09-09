// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline --legalize-stablehlo-composite-to-ttir -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["batch"=1, "vocab"=4]>

// CHECK-LABEL: func.func public @topk_multi_device
// CHECK: sdy.manual_computation
// CHECK: "ttir.topk"{{.*}}tensor<1x16384xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x128xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x128xi32>
// CHECK: "ttir.constant"{{.*}}tensor<1x128xi32>
// CHECK: "ttir.add"{{.*}}tensor<1x128xi32>
// CHECK: "ttir.topk"{{.*}}tensor<1x128xf32>
// CHECK: "ttir.gather"{{.*}}tensor<1x32xi32>
func.func public @topk_multi_device(
    %arg0: tensor<1x65536xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{}, {"vocab"}]>})
    -> (tensor<1x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<1x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 32 : i64,
                              largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl
  } : (tensor<1x65536xf32>) -> (tensor<1x32xf32>, tensor<1x32xi64>)
  return %0#0, %0#1 : tensor<1x32xf32>, tensor<1x32xi64>
}

// CHECK-LABEL: func.func public @topk_single_device
// CHECK: sdy.manual_computation
// CHECK: "ttir.topk"
// CHECK-NOT: ttir.all_gather
// CHECK: sdy.return
func.func public @topk_single_device(
    %arg0: tensor<1x65536xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>})
    -> (tensor<1x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>},
        tensor<1x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}) {
  %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 32 : i64,
                              largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl
  } : (tensor<1x65536xf32>) -> (tensor<1x32xf32>, tensor<1x32xi64>)
  return %0#0, %0#1 : tensor<1x32xf32>, tensor<1x32xi64>
}

// Regression: torch_xla emits redundant reshape(reshape(x, A), shape(x)) pairs
// around graph inputs that pass through composites. The lowering must walk
// through these pass-through reshapes to detect that the topk dim is sharded,
// otherwise it falls back to single-device topk and the per-chip outputs are
// never merged into a global top-K.
// CHECK-LABEL: func.func public @topk_multi_device_with_reshape
// CHECK: sdy.manual_computation
// CHECK: "ttir.topk"{{.*}}tensor<1x16384xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x128xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x128xi32>
// CHECK: "ttir.topk"{{.*}}tensor<1x128xf32>
func.func public @topk_multi_device_with_reshape(
    %arg0: tensor<1x65536xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{}, {"vocab"}]>})
    -> (tensor<1x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<1x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %r0 = stablehlo.reshape %arg0 : (tensor<1x65536xf32>) -> tensor<1x1x65536xf32>
  %r1 = stablehlo.reshape %r0 : (tensor<1x1x65536xf32>) -> tensor<1x65536xf32>
  %0:2 = stablehlo.composite "tenstorrent.topk" %r1 {
      composite_attributes = {dim = -1 : i64, k = 32 : i64,
                              largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl
  } : (tensor<1x65536xf32>) -> (tensor<1x32xf32>, tensor<1x32xi64>)
  return %0#0, %0#1 : tensor<1x32xf32>, tensor<1x32xi64>
}

// Regression: matmul → composite case. The topk's operand is the result of
// stablehlo.dot_general whose RHS is column-sharded. The lowering must walk
// back through dot_general (mapping output dim to RHS non-contracting dim)
// and transpose (inverting the permutation) to discover the underlying
// sharded block arg, otherwise it falls back to single-device topk.
// CHECK-LABEL: func.func public @topk_multi_device_with_matmul
// CHECK: sdy.manual_computation
// CHECK: "ttir.topk"{{.*}}tensor<8x16384xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<8x128xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<8x128xi32>
// CHECK: "ttir.topk"{{.*}}tensor<8x128xf32>
func.func public @topk_multi_device_with_matmul(
    %arg0: tensor<8x2048xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg1: tensor<65536x2048xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{"vocab"}, {}]>})
    -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<8x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // weight^T: (vocab, hidden) -> (hidden, vocab)
  %t = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<65536x2048xf32>) -> tensor<2048x65536xf32>
  // logits = hidden @ weight^T
  %d = stablehlo.dot_general %arg0, %t, contracting_dims = [1] x [0]
      : (tensor<8x2048xf32>, tensor<2048x65536xf32>) -> tensor<8x65536xf32>
  %0:2 = stablehlo.composite "tenstorrent.topk" %d {
      composite_attributes = {dim = -1 : i64, k = 32 : i64,
                              largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl_matmul
  } : (tensor<8x65536xf32>) -> (tensor<8x32xf32>, tensor<8x32xi64>)
  return %0#0, %0#1 : tensor<8x32xf32>, tensor<8x32xi64>
}

func.func private @tenstorrent.topk.impl_matmul(%arg0: tensor<8x65536xf32>)
    -> (tensor<8x32xf32>, tensor<8x32xi64>) {
  %0 = stablehlo.iota dim = 0 : tensor<65536xi32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<65536xi32>) -> tensor<8x65536xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
    %cmp = stablehlo.compare GT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<8x65536xf32>, tensor<8x65536xi32>) -> (tensor<8x65536xf32>, tensor<8x65536xi32>)
  %3 = stablehlo.slice %2#0 [0:8, 0:32] : (tensor<8x65536xf32>) -> tensor<8x32xf32>
  %4 = stablehlo.slice %2#1 [0:8, 0:32] : (tensor<8x65536xi32>) -> tensor<8x32xi32>
  %5 = stablehlo.convert %4 : (tensor<8x32xi32>) -> tensor<8x32xi64>
  return %3, %5 : tensor<8x32xf32>, tensor<8x32xi64>
}

func.func private @tenstorrent.topk.impl(%arg0: tensor<1x65536xf32>)
    -> (tensor<1x32xf32>, tensor<1x32xi64>) {
  %0 = stablehlo.iota dim = 0 : tensor<65536xi32>
  %1 = stablehlo.reshape %0 : (tensor<65536xi32>) -> tensor<1x65536xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
    %cmp = stablehlo.compare GT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<1x65536xf32>, tensor<1x65536xi32>) -> (tensor<1x65536xf32>, tensor<1x65536xi32>)
  %3 = stablehlo.slice %2#0 [0:1, 0:32] : (tensor<1x65536xf32>) -> tensor<1x32xf32>
  %4 = stablehlo.slice %2#1 [0:1, 0:32] : (tensor<1x65536xi32>) -> tensor<1x32xi32>
  %5 = stablehlo.convert %4 : (tensor<1x32xi32>) -> tensor<1x32xi64>
  return %3, %5 : tensor<1x32xf32>, tensor<1x32xi64>
}

// Regression: K=1 (greedy) on a sharded input. k*numShards = 4 is below the
// tt-metal tile width (32), so the lowering pads each shard's local topk to a
// full tile (effectiveK=32) to keep the all_gather'd intermediate tile-aligned,
// then slices down to k=1 at the end.
// CHECK-LABEL: func.func public @topk_indices_k1_multi_device
// CHECK: sdy.manual_computation
// CHECK: "ttir.topk"{{.*}}k = 32 : i32{{.*}}tensor<1x16384xf32>{{.*}}tensor<1x32xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x32xf32>{{.*}}tensor<1x128xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x32xi32>{{.*}}tensor<1x128xi32>
// CHECK: "ttir.add"{{.*}}tensor<1x128xi32>
// CHECK: "ttir.topk"{{.*}}k = 32 : i32{{.*}}tensor<1x128xf32>{{.*}}tensor<1x32xf32>
// CHECK: "ttir.gather"{{.*}}tensor<1x128xi32>{{.*}}tensor<1x32xi32>
// CHECK: "ttir.slice_static"{{.*}}tensor<1x32xi32>{{.*}}tensor<1x1xi32>
func.func public @topk_indices_k1_multi_device(
    %arg0: tensor<1x65536xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{}, {"vocab"}]>})
    -> (tensor<1x1xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.composite "tenstorrent.topk_indices" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 1 : i64,
                              largest = true, sorted = false},
      decomposition = @tenstorrent.topk_indices.impl_k1
  } : (tensor<1x65536xf32>) -> tensor<1x1xi64>
  return %0 : tensor<1x1xi64>
}

func.func private @tenstorrent.topk_indices.impl_k1(%arg0: tensor<1x65536xf32>)
    -> tensor<1x1xi64> {
  %0 = stablehlo.iota dim = 0 : tensor<65536xi32>
  %1 = stablehlo.reshape %0 : (tensor<65536xi32>) -> tensor<1x65536xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
    %cmp = stablehlo.compare GT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<1x65536xf32>, tensor<1x65536xi32>) -> (tensor<1x65536xf32>, tensor<1x65536xi32>)
  %4 = stablehlo.slice %2#1 [0:1, 0:1] : (tensor<1x65536xi32>) -> tensor<1x1xi32>
  %5 = stablehlo.convert %4 : (tensor<1x1xi32>) -> tensor<1x1xi64>
  return %5 : tensor<1x1xi64>
}
