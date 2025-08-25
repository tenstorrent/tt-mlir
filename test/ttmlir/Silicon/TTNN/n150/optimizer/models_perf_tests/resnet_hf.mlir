// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-pass=true" -o resnet_hf_ttnn.mlir %models/resnet_hf.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet_hf_ttnn.mlir
// RUN: ttrt run %t.ttnn
