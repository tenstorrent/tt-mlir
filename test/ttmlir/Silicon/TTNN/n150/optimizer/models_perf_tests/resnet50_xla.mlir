// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" -o resnet50_xla_ttnn.mlir %models/resnet50_xla.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet50_xla_ttnn.mlir
// RUN: ttrt run %t.ttnn
// UNSUPPORTED: true
// Test failing due to DRAM out of memory errors during buffer allocation
