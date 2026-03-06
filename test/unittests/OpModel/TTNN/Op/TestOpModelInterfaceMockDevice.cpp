// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test binary runs in its own process so that MetalContext is initialized
// with a mock topology from the start, without being polluted by a
// real-hardware topology from other test binaries.
//
// Mock mode is configured once (SetUpTestSuite) with maxMockChips chips.
// Individual tests reshape the MeshDevice to the desired topology via
// reshapeMeshDevice(), which destroys and recreates only the MeshDevice
// without cycling configure_mock_mode/disable_mock_mode — that cycle
// crashes in the same process (ethernet core timeout).

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <cstdint>

namespace mlir::tt::ttnn {

// Maximum number of mock chips configured once per binary.
static constexpr size_t maxMockChips = 8;

// Param: {meshRows, meshCols, dim, clusterAxis}
struct MeshPartitionParam {
  size_t meshRows;
  size_t meshCols;
  int32_t dim;
  uint32_t clusterAxis;
};

// Fixture that configures mock mode once (SetUpTestSuite) and reshapes the
// mesh device per test. Inherits helpers from OpModelFixture.
class OpModelMockMeshInterfaceTest
    : public OpModelFixture,
      public ::testing::WithParamInterface<MeshPartitionParam> {
public:
  // Configure mock mode once for the entire test suite.
  static void SetUpTestSuite() {
    // Create a temporary MLIRContext just to build the initial system desc.
    mlir::MLIRContext tmpCtx;
    tmpCtx.loadDialect<ttcore::TTCoreDialect>();
    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &tmpCtx, ttcore::Arch::WormholeB0, {1, maxMockChips});
    op_model::SingletonDeviceContext::setSystemDesc(systemDesc);
    op_model::SingletonDeviceContext::getInstance().openMockDevice(
        /*traceRegionSize=*/0,
        /*meshShape=*/std::make_pair(static_cast<size_t>(1), maxMockChips));
  }

  static void TearDownTestSuite() {
    if (op_model::SingletonDeviceContext::getInstance().isDeviceInitialized()) {
      op_model::SingletonDeviceContext::getInstance().closeInstance();
    }
  }

  void SetUp() override {
    const auto &p = GetParam();

    // Initialize MLIR context and module.
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());

    // Update system desc and reshape the mock device for this test's topology.
    auto systemDesc = ttcore::SystemDescAttr::getDefault(
        &context, ttcore::Arch::WormholeB0,
        {static_cast<int>(p.meshRows), static_cast<int>(p.meshCols)});
    op_model::SingletonDeviceContext::setSystemDesc(systemDesc);
    op_model::SingletonDeviceContext::getInstance().reshapeMeshDevice(
        {p.meshRows, p.meshCols});

    mlir::tt::ttcore::registerDevice(module.get());
  }

  void TearDown() override {
    // Don't close the device — reshapeMeshDevice will handle it next SetUp.
    // TearDownTestSuite does the final close.
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

TEST_P(OpModelMockMeshInterfaceTest, MeshPartitionOpInterface) {
  const auto &p = GetParam();
  const size_t meshShape[] = {p.meshRows, p.meshCols};
  const int64_t splitFactor = static_cast<int64_t>(meshShape[p.clusterAxis]);

  llvm::SmallVector<int64_t> inputShape = {8, 1024};
  llvm::SmallVector<int64_t> outputShape = inputShape;
  outputShape[p.dim] = inputShape[p.dim] / splitFactor;

  auto inputLayout = CreateRowMajorLayout(inputShape, BufferType::DRAM,
                                          TensorMemoryLayout::Interleaved);
  auto input =
      createEmptyTensor(inputShape, builder.getBF16Type(), inputLayout);
  auto outputType = createRankedTensorType(outputShape);

  auto meshPartitionOp = builder.create<MeshPartitionOp>(
      builder.getUnknownLoc(), outputType, input,
      /*dim=*/builder.getSI32IntegerAttr(p.dim),
      /*cluster_axis=*/builder.getUI32IntegerAttr(p.clusterAxis),
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

INSTANTIATE_TEST_SUITE_P(
    MeshPartition, OpModelMockMeshInterfaceTest,
    ::testing::Values(
        // {1,8} mesh: axis 0=1, axis 1=8. Only axis 1 splits.
        MeshPartitionParam{1, 8, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 8, 1, 1}, // split dim 1 on axis 1
        // {8,1} mesh: axis 0=8, axis 1=1. Only axis 0 splits.
        MeshPartitionParam{8, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{8, 1, 1, 0}, // split dim 1 on axis 0
        // {2,4} mesh: both axes split (axis 0=2, axis 1=4).
        MeshPartitionParam{2, 4, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{2, 4, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{2, 4, 1, 0}, // split dim 1 on axis 0
        MeshPartitionParam{2, 4, 1, 1}, // split dim 1 on axis 1
        // {4,2} mesh: both axes split (axis 0=4, axis 1=2).
        MeshPartitionParam{4, 2, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{4, 2, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{4, 2, 1, 0}, // split dim 1 on axis 0
        MeshPartitionParam{4, 2, 1, 1}, // split dim 1 on axis 1
        // {1,2} mesh: axis 0=1, axis 1=2. Only axis 1 splits.
        MeshPartitionParam{1, 2, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 2, 1, 1}, // split dim 1 on axis 1
        // {2,1} mesh: axis 0=2, axis 1=1. Only axis 0 splits.
        MeshPartitionParam{2, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{2, 1, 1, 0}, // split dim 1 on axis 0
        // {1,4} mesh: axis 0=1, axis 1=4. Only axis 1 splits.
        MeshPartitionParam{1, 4, 0, 1}, // split dim 0 on axis 1
        MeshPartitionParam{1, 4, 1, 1}, // split dim 1 on axis 1
        // {4,1} mesh: axis 0=4, axis 1=1. Only axis 0 splits.
        MeshPartitionParam{4, 1, 0, 0}, // split dim 0 on axis 0
        MeshPartitionParam{4, 1, 1, 0}  // split dim 1 on axis 0
        ));

} // namespace mlir::tt::ttnn
