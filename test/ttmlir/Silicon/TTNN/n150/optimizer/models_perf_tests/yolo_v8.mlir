// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false enable-fusing-pass=true" -o yolo_v8_ttnn.mlir %models/yolo_v8.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer yolo_v8_ttnn.mlir > %t.ttnn
// RUN: ttrt run %t.ttnn
