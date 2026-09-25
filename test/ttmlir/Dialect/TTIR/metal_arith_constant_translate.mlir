// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=PIPELINE
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

// Regression test for https://github.com/tenstorrent/tt-mlir/issues/8722
// TTMetal flatbuffer translator must register the arith dialect: scalar
// arguments materialized via host-layout copies lower to arith.constant
// (e.g. index constants for scalar buffer accesses), and translation must
// not reject them.
module @jit_foo attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  ttcore.device_module {
    builtin.module @jit_foo attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<16x22xf64>, %arg3: tensor<22x18xf64>, %arg4: tensor<18x24xf64>, %arg5: tensor<16x24xf64>) -> tensor<16x24xf64> {
        // PIPELINE: arith.constant
        %0 = "ttir.dot_general"(%arg2, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<16x22xf64>, tensor<22x18xf64>) -> tensor<16x18xf64>
        %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32]}> : (tensor<f64>) -> tensor<1x1xf64>
        %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 16, 18>}> : (tensor<1x1xf64>) -> tensor<16x18xf64>
        %3 = "ttir.multiply"(%0, %2) : (tensor<16x18xf64>, tensor<16x18xf64>) -> tensor<16x18xf64>
        %4 = "ttir.dot_general"(%3, %arg4) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<16x18xf64>, tensor<18x24xf64>) -> tensor<16x24xf64>
        %5 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32]}> : (tensor<f64>) -> tensor<1x1xf64>
        %6 = "ttir.broadcast"(%5) <{broadcast_dimensions = array<i64: 16, 24>}> : (tensor<1x1xf64>) -> tensor<16x24xf64>
        %7 = "ttir.multiply"(%arg5, %6) : (tensor<16x24xf64>, tensor<16x24xf64>) -> tensor<16x24xf64>
        %8 = "ttir.add"(%4, %7) : (tensor<16x24xf64>, tensor<16x24xf64>) -> tensor<16x24xf64>
        return %8 : tensor<16x24xf64>
      }
    }
  }
}
