// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TEST_UNITTESTS_MLIRATTRTOFBNATIVE_OPTPATHPARITY_H
#define TTMLIR_TEST_UNITTESTS_MLIRATTRTOFBNATIVE_OPTPATHPARITY_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/FuncOpToProgram.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "flatbuffers/flatbuffer_builder.h"
#include "gtest/gtest.h"

#include <memory>
#include <optional>

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::TensorRef>
tensorValueToFlatbuffer(::mlir::tt::FlatbufferObjectCache &cache,
                        ::mlir::Value value,
                        ::mlir::tt::ttcore::ShardStatus shardStatus,
                        std::optional<::mlir::RankedTensorType> localShape);
} // namespace mlir::tt::ttnn

struct Env {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder;

  Env() : builder(&context) {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
  }
};

inline Env &env() {
  static Env e;
  return e;
}

inline mlir::MLIRContext *getContext() { return &env().context; }

inline mlir::tt::ttnn::TTNNLayoutAttr
createTiledL1InterleavedLayout(llvm::ArrayRef<int64_t> shape) {
  auto &e = env();
  auto device = mlir::tt::ttcore::lookupDevice(e.module.get());
  auto tileType = mlir::tt::ttcore::TileType::get(e.builder.getBF16Type());
  llvm::SmallVector<int64_t> gridShape = {1, 1, 32, 32};
  return mlir::tt::ttnn::TTNNLayoutAttr::Builder(&e.context, shape, tileType)
      .setBufferType(mlir::tt::ttnn::BufferType::L1)
      .setMemoryLayout(mlir::tt::ttnn::TensorMemoryLayout::Interleaved)
      .setGridShape(gridShape)
      .buildWithCanonicalCorePlacement(device);
}

inline mlir::RankedTensorType tiledL1BF16Type(llvm::ArrayRef<int64_t> shape) {
  return mlir::RankedTensorType::get(shape, env().builder.getBF16Type(),
                                     createTiledL1InterleavedLayout(shape));
}

template <typename OpT>
inline mlir::tt::ttnn::TTNNLayoutAttr resolveOutputLayout(OpT op) {
  return mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
      mlir::cast<mlir::RankedTensorType>(op.getResult().getType())
          .getEncoding());
}

template <typename... Values>
inline void prepopulateOperandTensorRefs(mlir::tt::FlatbufferObjectCache &cache,
                                         Values... operands) {
  (cache.getOrCreateNoSharding(
       mlir::tt::ttnn::getOperandThroughDPSOps(operands),
       mlir::tt::ttnn::tensorValueToFlatbuffer,
       /*localShape=*/std::nullopt),
   ...);
}

#endif // TTMLIR_TEST_UNITTESTS_MLIRATTRTOFBNATIVE_OPTPATHPARITY_H
