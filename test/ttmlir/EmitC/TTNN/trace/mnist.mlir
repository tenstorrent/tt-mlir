UNSUPPORTED: true
// https://github.com/tenstorrent/tt-mlir/issues/4772

RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-trace=true" -o %t.mlir %models/mnist.mlir
RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc -o %t2.mlir %t.mlir
RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
