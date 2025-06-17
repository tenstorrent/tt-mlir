// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn_to_emitpy::utils {

mlir::IntegerAttr createIndex(Builder &builder, int64_t idx) {
  return mlir::IntegerAttr::get(builder.getIndexType(), idx);
}

emitpy::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr) {
  llvm::ArrayRef shape = attr.getShape();
  std::string buf;
  llvm::raw_string_ostream rso(buf);

  llvm::interleaveComma(shape, rso);

  return builder.getType<emitpy::OpaqueAttr>("ttnn.Shape([" + rso.str() + "])");
}

emitpy::CallOpaqueOp createShapeOp(ConversionPatternRewriter &rewriter,
                                   ttnn::ShapeAttr shapeAttr, Location loc) {
  llvm::StringRef shapeNameStr = "ttnn.Shape";
  return rewriter.create<emitpy::CallOpaqueOp>(
      loc, emitpy::OpaqueType::get(rewriter.getContext(), shapeNameStr),
      shapeNameStr, rewriter.getArrayAttr(convertShape(rewriter, shapeAttr)),
      ValueRange());
}

emitpy::OpaqueAttr convertLayoutAttr(Builder &builder, ttnn::LayoutAttr attr) {
  switch (attr.getValue()) {
  case ttnn::Layout::RowMajor:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.Layout.ROW_MAJOR");
  case ttnn::Layout::Tile:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.Layout.TILE");
  case ttnn::Layout::Invalid:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.Layout.INVALID");
  }

  llvm_unreachable("Unknown ttnn.Layout");
}

emitpy::OpaqueAttr convertBufferType(Builder &builder,
                                     ttnn::BufferTypeAttr attr) {
  switch (attr.getValue()) {
  case ttnn::BufferType::DRAM:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.BufferType.DRAM");
  case ttnn::BufferType::L1:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.BufferType.L1");
  case ttnn::BufferType::L1Small:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.BufferType.L1_SMALL");
  case ttnn::BufferType::SystemMemory:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.BufferType.SYSTEM_MEMORY");
  case ttnn::BufferType::Trace:
    return builder.getType<emitpy::OpaqueAttr>("ttnn.BufferType.TRACE");
  }

  llvm_unreachable("Unknown ttnn.BufferType");
}

} // namespace mlir::tt::ttnn_to_emitpy::utils
