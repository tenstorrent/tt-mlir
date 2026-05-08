// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m {

Value getOrCreateCB(RewriterBase &rewriter, GenericOp generic, Block *block,
                    unsigned cbOperandIndex) {
  // If CB already exists, return it
  Value cb = nullptr;
  block->walk([&](GetCBOp getCBOp) {
    if (getCBOp.getCbOperandIdx() == cbOperandIndex) {
      cb = getCBOp.getResult();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (cb) {
    return cb;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(block);
  Value localBuffer = generic.getOperand(cbOperandIndex);
  cb = rewriter
           .create<GetCBOp>(
               generic.getLoc(),
               CBType::get(rewriter.getContext(),
                           mlir::cast<ShapedType>(localBuffer.getType())),
               cbOperandIndex,
               ResolutionStageAttr::get(rewriter.getContext(),
                                        ResolutionStage::Compile))
           .getResult();
  return cb;
}

} // namespace mlir::tt::d2m
