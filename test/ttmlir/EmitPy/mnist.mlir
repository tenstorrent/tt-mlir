// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %models/mnist.mlir
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
