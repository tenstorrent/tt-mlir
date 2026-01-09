// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttnn {

class OpModelTest : public OpModelFixture {
public:
  mlir::RankedTensorType
  createRankedTensorType(llvm::ArrayRef<int64_t> shape,
                         mlir::Type elementType = nullptr,
                         TTNNLayoutAttr layout = nullptr) {
    if (!elementType) {
      elementType = builder.getBF16Type();
    }
    return RankedTensorType::get(shape, elementType, layout);
  }

  mlir::Value createEmptyTensor(llvm::ArrayRef<int64_t> tensorShape,
                                mlir::Type elementType = nullptr,
                                TTNNLayoutAttr layout = nullptr) {
    if (!elementType) {
      elementType = builder.getBF16Type();
    }
    RankedTensorType rankedTensorType =
        createRankedTensorType(tensorShape, elementType, layout);
    return builder.create<OnesOp>(
        builder.getUnknownLoc(), rankedTensorType, nullptr,
        ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);
  }
};

// Test that bypassing the check in MatmulOp::getOpConstraints causes a crash.
// The check prevents crashes by returning early for block-sharded matmul inputs
// with logical shape less than tile size.
//
// IMPORTANT: If this test FAILS (EXPECT_DEATH fails because code doesn't
// crash), it means the metal issue
// (https://github.com/tenstorrent/tt-metal/pull/33777) is fixed and uplifted.
// In that case, all changes from this PR should be reverted:
// https://github.com/tenstorrent/tt-mlir/pull/6169
TEST_F(OpModelTest, MatmulBlockShardedInputWithPadding) {
  llvm::SmallVector<int64_t> inputShapeA = {4096, 16};
  TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, BufferType::L1, TensorMemoryLayout::BlockSharded,
      llvm::SmallVector<int64_t>{8, 1});

  llvm::SmallVector<int64_t> inputShapeB = {16, 128};
  TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto inputA = createEmptyTensor(inputShapeA, nullptr, inputLayoutA);
  auto inputB = createEmptyTensor(inputShapeB, nullptr, inputLayoutB);

  llvm::SmallVector<int64_t> outputShape = {4096, 128};
  auto outputType =
      createRankedTensorType(outputShape, builder.getBF16Type(), nullptr);

  auto matmul = builder.create<MatmulOp>(builder.getUnknownLoc(), outputType,
                                         mlir::ValueRange{inputA, inputB});

  auto deviceGrid = CreateWorkerGrid();

  EXPECT_DEATH(
      {
        auto constraintsExp = op_model::OpModel<MatmulOp>::getOpConstraints(
            deviceGrid, inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
            nullptr, matmul.getTransposeA(), matmul.getTransposeB());
        // Don't check the error - this will crash if fix is not in place
        (void)constraintsExp;
      },
      ".*"); // Match any crash message
}

// Test that matmul with activation and IgnorePhysicalLayout on sharded output
// crashes in tt-metal. The workaround in
// ShardSolver::shouldUseIgnorePhysicalLayout prevents this by using full layout
// when activation is present.
//
// IMPORTANT: If this test FAILS (EXPECT_DEATH fails because code doesn't
// crash), it means the metal issue (tt-metal#34500) is fixed and uplifted.
// In that case, the activation check in shouldUseIgnorePhysicalLayout can be
// removed from: lib/Dialect/TTNN/Analysis/ShardSolver.cpp
TEST_F(OpModelTest, MatmulActivationWithIgnorePhysicalLayout) {
  // Create interleaved (non-sharded) input tensors
  llvm::SmallVector<int64_t> inputShapeA = {64, 128};
  TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  llvm::SmallVector<int64_t> inputShapeB = {128, 64};
  TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto inputA = createEmptyTensor(inputShapeA, nullptr, inputLayoutA);
  auto inputB = createEmptyTensor(inputShapeB, nullptr, inputLayoutB);

  llvm::SmallVector<int64_t> outputShape = {64, 64};
  auto outputType =
      createRankedTensorType(outputShape, builder.getBF16Type(), nullptr);

  // Create matmul with activation attribute
  auto matmul = builder.create<MatmulOp>(builder.getUnknownLoc(), outputType,
                                         mlir::ValueRange{inputA, inputB});
  matmul.setActivation(llvm::StringRef("sigmoid"));

  // Create sharded output layout with IgnorePhysicalLayout=true
  auto outputLayout = CreateTiledLayout(outputShape, BufferType::L1,
                                        TensorMemoryLayout::BlockSharded,
                                        llvm::SmallVector<int64_t>{2, 2})
                          .withIgnorePhysicalLayout(true);

  auto deviceGrid = CreateWorkerGrid();

  // This crashes with "bad optional access" because tt-metal cannot handle
  // partial memory configs (IgnorePhysicalLayout=true) with fused activations.
  EXPECT_DEATH(
      {
        auto constraintsExp = op_model::OpModel<MatmulOp>::getOpConstraints(
            deviceGrid, inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
            outputLayout, matmul.getTransposeA(), matmul.getTransposeB(),
            matmul.getActivation());
        (void)constraintsExp;
      },
      ".*"); // Match any crash message
}

} // namespace mlir::tt::ttnn
