// This test isn't intended as an llvm-lit test. It's part of the CI job. You can test it locally by running (assuming your working directory is the root of the project):
// source env/activate
// cd test/ttmlir/EmitC/TTNN/load_input/load_input_local
// tt-alchemist generate-cpp --pipeline-options 'load-input-tensors-from-disk=true' ttnn_load_input_tensors.mlir --output load_input_cpp --standalone
// cp arg0.tensorbin arg1.tensorbin load_input_cpp
// cd !$
// ./run
//
// `arg0.tensorbin` and `arg1.tensorbin` are serialized `ttnn.full(3)` and `ttnn.full(5)`, respectively

module {
  func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
