// REQUIRES: opmodel, perf
// RUN: mkdir -p %t
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true enable-fusing-conv2d-with-multiply-pattern=true" -o %t/ttnn.mlir %models/resnet_hf.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer  %t/ttnn.mlir > %t/out.ttnn
// RUN: ttrt run %t/out.ttnn --result-file %t/run_results.json --benchmark 10.5
