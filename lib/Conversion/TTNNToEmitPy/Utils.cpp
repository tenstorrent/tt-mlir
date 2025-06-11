// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/Utils.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn_to_emitpy::utils {

emitpy::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr) {
  llvm::ArrayRef shape = attr.getShape();
  std::string buf;
  llvm::raw_string_ostream rso(buf);

  llvm::interleaveComma(shape, rso);

  return builder.getType<emitpy::OpaqueAttr>("{" + rso.str() + "}");
}

emitpy::CallOpaqueOp createShapeOp(ConversionPatternRewriter &rewriter,
                                   ttnn::ShapeAttr shapeAttr, Location loc) {
  llvm::StringRef shapeTypeStr = "ttnn.Shape";

  return rewriter.create<emitc::CallOpaqueOp>(
      loc, emitc::OpaqueType::get(rewriter.getContext(), shapeTypeStr),
      rewriter.getStringAttr(shapeTypeStr),
      rewriter.getArrayAttr(convertShape(rewriter, shapeAttr)), nullptr,
      ValueRange());
}

} // namespace mlir::tt::ttnn_to_emitpy::utils
