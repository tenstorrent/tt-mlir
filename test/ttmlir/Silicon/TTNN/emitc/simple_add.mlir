// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: ttmlir-opt --convert-ttnn-to-emitc %t.mlir > %emitc.mlir
// RUN: ttmlir-translate --mlir-to-cpp %emitc.mlir --allow-unregistered-dialect > %emitted.cpp
// RUN: cat %emitted.cpp >| /localdev/svuckovic/_workspace/repos/tt-mlir/tools/ttnn-standalone/ttnn-dylib.cpp
// RUN: ttrt run %t.ttnn --emitc-dylib /localdev/svuckovic/_workspace/repos/tt-mlir/tools/ttnn-standalone/build/libttnn-dylib.so --init ones

func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
