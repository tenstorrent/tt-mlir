// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o phi1_5_1layer_ttnn.mlir %models/phi1_5_1layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn phi1_5_1layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
