// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

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
