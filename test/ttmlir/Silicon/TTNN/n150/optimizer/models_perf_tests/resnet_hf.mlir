// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2" -o resnet_hf_ttnn.mlir %models/resnet_hf.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet_hf_ttnn.mlir
// RUN: ttrt run %t.ttnn
