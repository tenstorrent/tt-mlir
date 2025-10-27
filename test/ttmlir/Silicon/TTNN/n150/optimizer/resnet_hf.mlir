// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o resnet_hf_out.mlir %models/resnet_hf.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn resnet_hf_out.mlir
