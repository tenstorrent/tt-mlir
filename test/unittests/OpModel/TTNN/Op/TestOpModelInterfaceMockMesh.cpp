// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test binary runs in its own process so that MetalContext is initialized
// with a mock {1,8} mesh topology from the start, without being polluted by
// a real-hardware topology from other test binaries.

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace mlir::tt::ttnn {

// Fixture that opens a mock {1,8} mesh device from the start, so that
// MetalContext is initialized with the mock topology (not real hardware).
// Inherits helpers from OpModelFixture and adds OpModelBase-style helpers.
class OpModelMockMeshInterfaceTest : public OpModelFixture {
public:
  void SetUp() override {
    // Initialize MLIR context and module without opening a real device.
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());

    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &context, ttcore::Arch::WormholeB0, {1, 8});
    op_model::SingletonDeviceContext::setSystemDesc(systemDesc);
    op_model::SingletonDeviceContext::getInstance().openMockDevice(
        /*traceRegionSize=*/0,
        /*meshShape=*/std::make_pair<size_t, size_t>(1, 8));

    mlir::tt::ttcore::registerDevice(module.get());
  }

  llvm::Expected<op_model::OpConstraints> getOpConstraints(Operation *op) {
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      return backend.getOpConstraints(getInputLayouts(op), getOutputLayout(op));
    }
    return llvm::createStringError("Could not cast op to OpModel");
  }

  std::vector<TTNNLayoutAttr> getInputLayouts(Operation *op) {
    std::vector<TTNNLayoutAttr> inputs;

    auto limit = op->getNumOperands();
    if (isa<DestinationStyleOpInterface>(op)) {
      limit--;
    }

    for (size_t i = 0; i < limit; i++) {
      auto operand = op->getOperand(i);
      if (!operand || !mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      auto operandType = mlir::cast<RankedTensorType>(operand.getType());
      if (operandType.getEncoding()) {
        inputs.push_back(mlir::cast<TTNNLayoutAttr>(operandType.getEncoding()));
        continue;
      }

      auto inputShape = operandType.getShape();
      auto inputLayout = CreateTiledLayout(inputShape, BufferType::L1,
                                           TensorMemoryLayout::Interleaved);
      inputs.push_back(inputLayout);
    }
    return inputs;
  }

  TTNNLayoutAttr getOutputLayout(Operation *op) {
    auto output = op->getResult(0);
    auto outputShape =
        mlir::cast<RankedTensorType>(output.getType()).getShape();
    return CreateTiledLayout(outputShape, BufferType::L1,
                             TensorMemoryLayout::Interleaved);
  }

  ttcore::DeviceAttr getFakeDeviceAttr() {
    auto deviceIdx = mlir::getAffineConstantExpr(0, &context);
    auto shardOffset = mlir::getAffineConstantExpr(0, &context);
    auto d0 = mlir::getAffineDimExpr(0, &context);
    auto d1 = mlir::getAffineDimExpr(1, &context);
    auto map3 = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1}, &context);
    auto map4 = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1, shardOffset},
        &context);
    auto workerGrid = ttcore::GridAttr::get(&context, gridShapeHwN300, map3);

    return ttcore::DeviceAttr::get(&context, workerGrid, map4, map4, {1}, {0},
                                   {});
  }

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

TEST_F(OpModelMockMeshInterfaceTest, MeshPartitionOpInterface) {
  llvm::SmallVector<int64_t> inputShape = {8, 1024};
  llvm::SmallVector<int64_t> outputShape = {1, 1024};

  auto inputLayout = CreateRowMajorLayout(inputShape, BufferType::DRAM,
                                          TensorMemoryLayout::Interleaved);
  auto input =
      createEmptyTensor(inputShape, builder.getBF16Type(), inputLayout);
  auto outputType = createRankedTensorType(outputShape);

  auto meshPartitionOp = builder.create<MeshPartitionOp>(
      builder.getUnknownLoc(), outputType, input, builder.getSI32IntegerAttr(0),
      builder.getUI32IntegerAttr(1),
      /*memory_config=*/nullptr);
  meshPartitionOp->setAttr(ttcore::DeviceAttr::name, getFakeDeviceAttr());

  // test MeshPartitionOp interface - constraints
  auto constraintsExp = getOpConstraints(meshPartitionOp.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayouts] =
        l1;
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(totalPeakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
}

} // namespace mlir::tt::ttnn
