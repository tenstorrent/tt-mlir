// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

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
std::vector<ttnn::Tensor> utilCreateVec(T &&...t) {
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
  std::ostringstream oss;
  std::copy(shape.begin(), shape.end(), std::ostream_iterator<int>(oss, ", "));
  return builder.getType<emitc::OpaqueAttr>("{" + oss.str() + "}");
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
  case ttnn::TensorMemoryLayout::SingleBank:
    return builder.getType<emitc::OpaqueAttr>(
        "ttnn::TensorMemoryLayout::SINGLE_BANK");
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

emitc::OpaqueAttr convertDType(Builder &builder, tt::DataTypeAttr attr) {
  switch (attr.getValue()) {
  case tt::DataType::BFloat16:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT16");
  case tt::DataType::Float32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::FLOAT32");
  case tt::DataType::UInt32:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT32");
  case tt::DataType::BFP_BFloat8:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT8_B");
  case tt::DataType::BFP_BFloat4:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::BFLOAT4_B");
  case tt::DataType::UInt8:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT8");
  case tt::DataType::UInt16:
    return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::UINT16");
  // TODO(svuckovic):
  // Add support for INT32
  //
  // case tt::DataType::Int32:
  //   return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::INT32");
  case tt::DataType::Float16:
  case tt::DataType::BFP_Float2:
  case tt::DataType::BFP_Float4:
  case tt::DataType::BFP_Float8:
  case tt::DataType::BFP_BFloat2:
    llvm_unreachable("Unsupported ttnn::DataType");
  }

  llvm_unreachable("Unkonwn tt::DataType");
}

emitc::OpaqueAttr createStdNullopt(Builder &builder) {
  return builder.getType<emitc::OpaqueAttr>("std::nullopt");
}

emitc::ExpressionOp createShapeOp(ConversionPatternRewriter &rewriter,
                                  ttnn::ShapeAttr shapeAttr,
                                  Block *containingBlock, Location loc) {
  // Create ExpressionOp to hold multiple nested op calls, but will bundle them
  // together into a single SSA value
  //
  emitc::ExpressionOp shapeExpressionOp = rewriter.create<emitc::ExpressionOp>(
      loc, emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Shape"), false);

  // Add a block to the ExpressionOp, save current insertion point, and set
  // insertion point to newly added block
  //
  mlir::Block &bodyBlock = shapeExpressionOp.getBodyRegion().emplaceBlock();
  Block::iterator currentPoint = rewriter.getInsertionPoint();
  rewriter.setInsertionPointToStart(&bodyBlock);

  // Create a LegacyShape object
  //
  emitc::CallOpaqueOp metalShapeOp = rewriter.create<emitc::CallOpaqueOp>(
      loc,
      emitc::OpaqueType::get(rewriter.getContext(),
                             "tt::tt_metal::LegacyShape"),
      rewriter.getStringAttr("tt::tt_metal::LegacyShape"),
      rewriter.getArrayAttr(convertShape(rewriter, shapeAttr)), nullptr,
      ValueRange());

  // Create a ttnn::Shape object
  //
  emitc::CallOpaqueOp ttnnShapeOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, emitc::OpaqueType::get(rewriter.getContext(), "ttnn::Shape"),
      rewriter.getStringAttr("ttnn::Shape"), nullptr, nullptr,
      metalShapeOp->getResults());
  rewriter.create<emitc::YieldOp>(loc, ttnnShapeOp->getResult(0));

  // Reset to original insertion point
  //
  rewriter.setInsertionPoint(containingBlock, currentPoint);

  return shapeExpressionOp;
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
