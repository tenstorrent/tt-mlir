// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false enable-fusing-pass=true" -o segformer_ttnn.mlir %models/segformer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer segformer_ttnn.mlir > %t.ttnn
// RUN: ttrt run %t.ttnn
