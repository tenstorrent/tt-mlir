// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn_to_emitc::utils {

// Returns the closest parent module of the given operation
//
mlir::ModuleOp getParentModule(mlir::Operation *op) {
  while (op) {
    if (auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(op)) {
      return moduleOp;
    }
    op = op->getParentOp();
  }
  return nullptr;
}

// The func::FuncOp is inserted by creating an emitc::VerbatimOp with the
// function definition and inserting it at the start of the module.
//
bool insertVecCreateFnIfNotExists(PatternRewriter &rewriter, Operation *op) {
  ModuleOp moduleOp = getParentModule(op);
  assert(op && "Could not find top-level module");

  static constexpr const char *vecCreateFnAsStr = R"(
template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}
)";

  for (auto &currOp : moduleOp.getOps()) {
    if (auto verbatimOp = dyn_cast<emitc::VerbatimOp>(currOp)) {
      // Check if value of the VerbatimOp is the vecCreateFnAsStr - if so, it
      // means that the util vec function has already been added to the module
      //
      if (verbatimOp.getValue() == vecCreateFnAsStr) {
        return false;
      }
    }
  }

  // Set insertion to start of module, add the func there, and restore the
  // insertion point
  //
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(moduleOp.getBody(0));
  rewriter.create<emitc::VerbatimOp>(op->getLoc(), vecCreateFnAsStr);
  rewriter.restoreInsertionPoint(currentInsertionPoint);

  return true;
}

emitc::OpaqueAttr convertShape(Builder &builder, ttnn::ShapeAttr attr) {
  llvm::ArrayRef shape = attr.getShape();
  std::string buf;
  llvm::raw_string_ostream rso(buf);

  llvm::interleaveComma(shape, rso);

  return builder.getType<emitc::OpaqueAttr>("{" + rso.str() + "}");
}

emitc::OpaqueAttr convertTensorMemoryLayout(Builder &builder,
                                            ttnn::TensorMemoryLayoutAttr attr) {
  switch (attr.getValue()) {
  case ttnn::TensorMemoryLayout::BlockSharded:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::BLOCK_SHARDED");
  case ttnn::TensorMemoryLayout::HeightSharded:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::HEIGHT_SHARDED");
  case ttnn::TensorMemoryLayout::Interleaved:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::INTERLEAVED");
  case ttnn::TensorMemoryLayout::WidthSharded:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::WIDTH_SHARDED");
  }

  llvm_unreachable("Unknown ttnn::TensorMemoryLayout");
}

emitc::OpaqueAttr convertBufferType(Builder &builder,
                                    ttnn::BufferTypeAttr attr) {
  switch (attr.getValue()) {
  case ttnn::BufferType::DRAM:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::DRAM");
  case ttnn::BufferType::L1:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::L1");
  case ttnn::BufferType::L1Small:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::L1_SMALL");
  case ttnn::BufferType::SystemMemory:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::BufferType::SYSTEM_MEMORY");
  case ttnn::BufferType::Trace:
    return builder.getType<emitc::OpaqueAttr>("ttnn::BufferType::TRACE");
  }

  llvm_unreachable("Unknown ttnn::BufferType");
}

emitc::OpaqueAttr convertLayoutAttr(Builder &builder, ttnn::LayoutAttr attr) {
  switch (attr.getValue()) {
  case ttnn::Layout::RowMajor:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::ROW_MAJOR");
  case ttnn::Layout::Tile:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::TILE");
  case ttnn::Layout::Invalid:
    return builder.getType<emitc::OpaqueAttr>("ttnn::Layout::INVALID");
  }

  llvm_unreachable("Unknown ttnn::Layout");
}

emitc::OpaqueAttr convertBoolAttr(Builder &builder, BoolAttr attr) {
  return builder.getType<emitc::OpaqueAttr>(attr.getValue() ? "true" : "false");
}

emitc::OpaqueAttr convertDType(Builder &builder, ttcore::DataTypeAttr attr) {
  switch (attr.getValue()) {
  case ttcore::DataType::BFloat16:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT16");
  case ttcore::DataType::Float32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::FLOAT32");
  case ttcore::DataType::UInt32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT32");
  case ttcore::DataType::BFP_BFloat8:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT8_B");
  case ttcore::DataType::BFP_BFloat4:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT4_B");
  case ttcore::DataType::UInt8:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT8");
  case ttcore::DataType::UInt16:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT16");
  case ttcore::DataType::Int32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::INT32");
  case ttcore::DataType::Float16:
  case ttcore::DataType::BFP_Float2:
  case ttcore::DataType::BFP_Float4:
  case ttcore::DataType::BFP_Float8:
  case ttcore::DataType::BFP_BFloat2:
    llvm_unreachable("Unsupported ttnn::DataType");
  }

  llvm_unreachable("Unkonwn ttcore::DataType");
}

emitc::OpaqueAttr convertReduceType(ConversionPatternRewriter &rewriter,
                                    ttcore::ReduceType reduceType) {
  switch (reduceType) {
  case ttcore::ReduceType::Sum:
    return rewriter.getType<emitc::OpaqueAttr>(
        "::ttnn::operations::reduction::ReduceType::Sum");
  case ttcore::ReduceType::Mean:
    return rewriter.getType<emitc::OpaqueAttr>(
        "::ttnn::operations::reduction::ReduceType::Mean");
  case ttcore::ReduceType::Max:
    return rewriter.getType<emitc::OpaqueAttr>(
        "::ttnn::operations::reduction::ReduceType::Max");
  case ttcore::ReduceType::Min:
    return rewriter.getType<emitc::OpaqueAttr>(
        "::ttnn::operations::reduction::ReduceType::Min");
  case ttcore::ReduceType::Std:
    return rewriter.getType<emitc::OpaqueAttr>(
        "::ttnn::operations::reduction::ReduceType::Std");
  case ttcore::ReduceType::Var:
    return rewriter.getType<emitc::OpaqueAttr>(
        "::ttnn::operations::reduction::ReduceType::Var");
  }

  llvm_unreachable("Unknown ttnn::operations::reduction::ReduceType");
}

emitc::OpaqueAttr convertArrayAttrToTTNNSmallVector(Builder &builder,
                                                    ArrayAttr attr) {
  std::string buf;
  llvm::raw_string_ostream rso(buf);

  llvm::interleaveComma(attr, rso, [&](const Attribute &attr) {
    rso << mlir::cast<IntegerAttr>(attr).getInt();
  });

  return builder.getType<emitc::OpaqueAttr>("ttnn::SmallVector<int>{" +
                                            rso.str() + "}");
}

emitc::OpaqueAttr convertArrayAttrToSpan(Builder &builder, ArrayAttr attr) {
  std::string buf;
  llvm::raw_string_ostream rso(buf);

  llvm::interleaveComma(attr, rso, [&](const Attribute &attr) {
    rso << mlir::cast<IntegerAttr>(attr).getInt();
  });

  return builder.getType<emitc::OpaqueAttr>("std::vector<int>{" + rso.str() +
                                            "}");
}

emitc::OpaqueAttr createStdNullopt(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>("std::nullopt");
}

emitc::CallOpaqueOp createShapeOp(ConversionPatternRewriter &rewriter,
                                  ttnn::ShapeAttr shapeAttr, Location loc) {
  llvm::StringRef shapeTypeStr = "ttnn::Shape";

  return rewriter.create<emitc::CallOpaqueOp>(
      loc, emitc::OpaqueType::get(rewriter.getContext(), shapeTypeStr),
      rewriter.getStringAttr(shapeTypeStr),
      rewriter.getArrayAttr(convertShape(rewriter, shapeAttr)), nullptr,
      ValueRange());
}

emitc::CallOpaqueOp createMemoryConfigOp(ConversionPatternRewriter &rewriter,
                                         ttnn::MemoryConfigAttr memoryConfig,
                                         Location loc) {
  // Create ArrayAttr object holding MemoryConfig attributes
  //
  // TODO(svuckovic): (#620) Currently missing ShardSpec
  //
  ArrayAttr memCfgArrayAttrs = rewriter.getArrayAttr(
      {convertTensorMemoryLayout(rewriter,
                                 memoryConfig.getTensorMemoryLayout()),
       convertBufferType(rewriter, memoryConfig.getBufferType())});

  // Create MemoryConfig object
  //
  emitc::CallOpaqueOp memCfgOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, emitc::OpaqueType::get(rewriter.getContext(), "ttnn::MemoryConfig"),
      "ttnn::MemoryConfig", memCfgArrayAttrs, nullptr, ValueRange());

  return memCfgOp;
}

} // namespace mlir::tt::ttnn_to_emitc::utils
