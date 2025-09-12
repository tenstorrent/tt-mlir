// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
//
// This test tests multiple ops. It runs a single "ttir.add" op but it is expected that several other memory-related ops will be generated in the TTNN dialect:
// - ttnn.to_device
// - ttnn.to_layout
// - ttnn.deallocate
// - ttnn.from_device
//
// Line below checks that these ops appear after TTIR to TTNN conversion.
// RUN: FileCheck %s --input-file %t.mlir

func.func @multiple_ops(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = ttir.empty() : tensor<32x32xbf16>
  // CHECK: "ttnn.add"{{.*}}
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
