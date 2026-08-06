// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["batch"=2, "model"=4]>

// Batch dim is pass-through, so the kernel runs on the local rows (64 -> 32)
// and the result is batch sharded rather than gathered.
func.func @sampling_batch_sharded(%arg0: tensor<64x256xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg1: tensor<64x256xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg2: tensor<64xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}, %arg3: tensor<64xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}, %arg4: tensor<64xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}) -> tensor<64xi32> {
  %0 = stablehlo.custom_call @tt.sampling(%arg0, %arg1, %arg2, %arg3, %arg4) {api_version = 0 : i32, mhlo.frontend_attributes = {seed = "0"}} : (tensor<64x256xbf16>, tensor<64x256xi32>, tensor<64xi32>, tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

// CHECK-LABEL: func.func @sampling_batch_sharded
// CHECK: out_shardings=[<@mesh, [{"batch"}]>]
// CHECK: stablehlo.custom_call @tt.sampling
// CHECK-SAME: (tensor<32x256xbf16>, tensor<32x256xi32>, tensor<32xi32>, tensor<32xbf16>, tensor<32xbf16>) -> tensor<32xi32>

// The candidate dim needs replication: a row's candidate set must stay whole
// for softmax, top-k and multinomial. Sharding it on "model" is undone.
func.func @sampling_candidate_dim_replicated(%arg0: tensor<64x256xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg1: tensor<64x256xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg2: tensor<64xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}, %arg3: tensor<64xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}, %arg4: tensor<64xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}) -> tensor<64xi32> {
  %0 = stablehlo.custom_call @tt.sampling(%arg0, %arg1, %arg2, %arg3, %arg4) {api_version = 0 : i32, mhlo.frontend_attributes = {seed = "0"}} : (tensor<64x256xbf16>, tensor<64x256xi32>, tensor<64xi32>, tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

// CHECK-LABEL: func.func @sampling_candidate_dim_replicated
// The candidate dim is gathered back before the op, and only the batch dim
// stays sharded.
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.custom_call @tt.sampling
// CHECK-SAME: (tensor<32x256xbf16>, tensor<32x256xi32>, tensor<32xi32>, tensor<32xbf16>, tensor<32xbf16>) -> tensor<32xi32>
