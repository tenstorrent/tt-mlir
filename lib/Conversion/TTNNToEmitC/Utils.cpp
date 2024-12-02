// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"

#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttnntoemitc::utils {

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
  case ttnn::TensorMemoryLayout::None:
    llvm_unreachable("Unsupported ttnn::TensorMemoryLayout");
  }
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
  // TODO: (#620) Currently missing ShardSpec
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

} // namespace mlir::tt::ttnntoemitc::utils
