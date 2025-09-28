// This test isn't intended as an llvm-lit test. It's part of the CI job. You can test it locally by running (assuming your working directory is the root of the project):
// source env/activate
// cd test/ttmlir/EmitC/TTNN/load_input
// tt-alchemist generate-cpp --pipeline-options 'load-input-tensors-from-disk=true' ttnn_load_input_tensors.mlir --output load_input_cpp --standalone
// cp dump_inputs.py load_input_cpp
// cd !$
// python dump_inputs.py
// ./run

module {
  func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
