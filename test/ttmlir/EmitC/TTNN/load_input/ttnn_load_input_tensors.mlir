// This test isn't intended as an llvm-lit test. It's part of the CI job. You can test it locally by running (assuming your working directory is the root of the project):
// source env/activate
// cd test/ttmlir/EmitC/TTNN/load_input
// ttmlir-opt -ttir-to-emitc-pipeline="load-input-tensors-from-disk=true" ttnn_load_input_tensors.mlir | ttmlir-translate -mlir-to-cpp -o ttnn-standalone.cpp
// cp arg0.tensorbin arg1.tensorbin ttnn-standalone.cpp ../../../../../tools/ttnn-standalone
// cd !$
// ./run
//
// arg0.tensorbin and arg1.tensorbin represents serialized `ttnn.full(3)` and `ttnn.full(5), respectively

module {
  func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
