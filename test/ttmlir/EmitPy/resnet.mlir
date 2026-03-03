// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %models/resnet50_xla.mlir
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
