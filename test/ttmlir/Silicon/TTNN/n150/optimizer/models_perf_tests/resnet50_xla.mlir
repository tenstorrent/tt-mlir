// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2" -o resnet50_xla_ttnn.mlir %models/resnet50_xla.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_xla_ttnn.mlir
// RUN: ttrt run %t.ttnn
