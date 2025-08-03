// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpsModelCache.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {

namespace detail {
llvm::Expected<bool> checkDeviceWorkerGrid(mlir::Operation *op) {
  auto deviceAttr = ttcore::lookupDevice(op);
  assert(deviceAttr);
  return op_model::Device::getDeviceConstraints(deviceAttr.getWorkerGrid());
}

llvm::SmallVector<int64_t>
convertArrayAttrToSmallVec(mlir::ArrayAttr arrayAttr) {
  llvm::SmallVector<int64_t> result;
  for (const mlir::Attribute &attr : arrayAttr) {
    result.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
  }
  return result;
}

std::optional<llvm::SmallVector<int64_t>>
convertOptionalArrayAttrToSmallVec(std::optional<mlir::ArrayAttr> arrayAttr) {
  if (!arrayAttr.has_value()) {
    return std::nullopt;
  }
  return convertArrayAttrToSmallVec(arrayAttr.value());
}

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getUnaryOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, inputShape,
      inputs[0], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getUnaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                  const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(op_model::OpModel<OpT>::getOpRuntime, op,
                                       inputShape, inputs[0],
                                       opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getBinaryOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = op.getLhs().getType().getShape();
  const auto inputShapeB = op.getRhs().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, inputShapeA,
      inputs[0], inputShapeB, inputs[1], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getBinaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                   const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = op.getLhs().getType().getShape();
  const auto inputShapeB = op.getRhs().getType().getShape();

  return opRuntimeCache().getOrCompute(op_model::OpModel<OpT>::getOpRuntime, op,
                                       inputShapeA, inputs[0], inputShapeB,
                                       inputs[1], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getTernaryOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShapeA = op.getFirst().getType().getShape();
  const auto inputShapeB = op.getSecond().getType().getShape();
  const auto inputShapeC = op.getThird().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, inputShapeA,
      inputs[0], inputShapeB, inputs[1], inputShapeC, inputs[2],
      opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getTernaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShapeA = op.getFirst().getType().getShape();
  const auto inputShapeB = op.getSecond().getType().getShape();
  const auto inputShapeC = op.getThird().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<OpT>::getOpRuntime, op, inputShapeA, inputs[0],
      inputShapeB, inputs[1], inputShapeC, inputs[2], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getReductionOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  const auto inputShape = op.getInput().getType().getShape();
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, inputShape,
      inputs[0], detail::convertOptionalArrayAttrToSmallVec(op.getDimArg()),
      op.getKeepDim(), opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getReductionOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  const auto inputShape = op.getInput().getType().getShape();
  return opRuntimeCache().getOrCompute(
      op_model::OpModel<OpT>::getOpRuntime, op, inputShape, inputs[0],
      detail::convertOptionalArrayAttrToSmallVec(op.getDimArg()),
      op.getKeepDim(), opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getPoolingOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, inputShape,
      inputs[0], op.getBatchSize(), op.getInputHeight(), op.getInputWidth(),
      op.getChannels(), op.getKernelSize(), op.getStride(), op.getPadding(),
      op.getDilation(), op.getCeilMode(), op.getInPlaceHalo(),
      opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getPoolingOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<OpT>::getOpRuntime, op, inputShape, inputs[0],
      op.getBatchSize(), op.getInputHeight(), op.getInputWidth(),
      op.getChannels(), op.getKernelSize(), op.getStride(), op.getPadding(),
      op.getDilation(), op.getCeilMode(), op.getInPlaceHalo(),
      opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getNamedFullOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 0);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  const mlir::tt::ttnn::ShapeAttr shape = op.getShape();
  const std::optional<mlir::tt::ttcore::DataType> dtype = op.getDtype();
  const std::optional<mlir::tt::ttnn::Layout> layout = op.getLayout();
  const std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, shape, dtype,
      layout, memoryConfig, opConfig.outputLayout);
}
} // namespace detail

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ReluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
ReluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SqrtOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SqrtOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
SqrtOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SinOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SinOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
SinOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// AbsOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
AbsOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
AbsOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// CeilOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
CeilOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
CeilOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SignOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SignOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
SignOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// ErfOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ErfOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
ErfOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// ErfcOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ErfcOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
ErfcOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// FloorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
FloorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
FloorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// GeluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
GeluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
GeluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// IsFiniteOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
IsFiniteOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
IsFiniteOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LogicalNotOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogicalNotOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogicalNotOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// NegOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
NegOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
NegOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// TanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
TanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
TanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// AtanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
AtanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
AtanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// RsqrtOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RsqrtOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
RsqrtOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// Log1pOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
Log1pOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
Log1pOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// Expm1Op - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
Expm1Op::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
Expm1Op::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// CosOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
CosOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
CosOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// TanhOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
TanhOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
TanhOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LogOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// ReciprocalOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ReciprocalOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
ReciprocalOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// CbrtOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
CbrtOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
CbrtOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// BitwiseNotOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
BitwiseNotOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
BitwiseNotOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SigmoidOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SigmoidOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
SigmoidOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// ExpOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ExpOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
ExpOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LeakyReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LeakyReluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<LeakyReluOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getParameter(), opConfig.outputLayout);
}

llvm::Expected<size_t>
LeakyReluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<LeakyReluOp>::getOpRuntime, *this, inputShape,
      inputs[0], getParameter(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
AddOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
AddOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// MultiplyOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MultiplyOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MultiplyOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LogicalRightShiftOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogicalRightShiftOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                      const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogicalRightShiftOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                  const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SubtractOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SubtractOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
SubtractOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// MaximumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MaximumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MaximumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// MinimumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MinimumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MinimumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// DivideOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
DivideOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
DivideOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// EqualOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
EqualOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
EqualOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// NotEqualOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
NotEqualOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
NotEqualOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// GreaterEqualOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
GreaterEqualOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                 const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
GreaterEqualOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// GreaterThanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
GreaterThanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
GreaterThanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LessEqualOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LessEqualOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LessEqualOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LessThanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LessThanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LessThanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LogicalAndOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogicalAndOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogicalAndOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LogicalOrOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogicalOrOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogicalOrOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// LogicalXorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogicalXorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogicalXorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// WhereOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
WhereOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getTernaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
WhereOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getTernaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// MeanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MeanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return getReductionOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MeanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return getReductionOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getReductionOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
SumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return getReductionOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SoftmaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<SoftmaxOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDimension(), opConfig.outputLayout);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<SoftmaxOp>::getOpRuntime, *this, inputShape, inputs[0],
      getDimension(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ReshapeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ReshapeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ReshapeOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
ReshapeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ReshapeOp>::getOpRuntime, *this, inputShape, inputs[0],
      outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SliceStaticOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SliceStaticOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<SliceStaticOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], detail::convertArrayAttrToSmallVec(getBegins()),
      detail::convertArrayAttrToSmallVec(getEnds()),
      detail::convertArrayAttrToSmallVec(getStep()), opConfig.outputLayout);
}

llvm::Expected<size_t>
SliceStaticOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<SliceStaticOp>::getOpRuntime, *this, inputShape,
      inputs[0], detail::convertArrayAttrToSmallVec(getBegins()),
      detail::convertArrayAttrToSmallVec(getEnds()),
      detail::convertArrayAttrToSmallVec(getStep()), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// TypecastOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
TypecastOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<TypecastOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDtypeAttr(), opConfig.outputLayout);
}

llvm::Expected<size_t>
TypecastOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<TypecastOp>::getOpRuntime, *this, inputShape, inputs[0],
      getDtypeAttr(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ToLayoutOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ToLayoutOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  assert(opConfig.outputLayout.getLayout() == getLayout());

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ToLayoutOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDtype(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ToLayoutOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  assert(opConfig.outputLayout.getLayout() == getLayout());

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ToLayoutOp>::getOpRuntime, *this, inputShape, inputs[0],
      getDtype(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ToMemoryConfigOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                   const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ToMemoryConfigOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getMemoryConfig(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ToMemoryConfigOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ToMemoryConfigOp>::getOpRuntime, *this, inputShape,
      inputs[0], getMemoryConfig(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ConcatOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ConcatOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == getInputs().size());

  std::vector<llvm::ArrayRef<int64_t>> inputShapes;
  for (const Value &opInput : getInputs()) {
    mlir::RankedTensorType inputType =
        mlir::cast<mlir::RankedTensorType>(opInput.getType());
    inputShapes.push_back(inputType.getShape());
  }

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ConcatOp>::getOpConstraints, *this, deviceGrid,
      inputShapes, inputs, getDim(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ConcatOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == getInputs().size());

  std::vector<llvm::ArrayRef<int64_t>> inputShapes;
  for (const Value &opInput : getInputs()) {
    mlir::RankedTensorType inputType =
        mlir::cast<mlir::RankedTensorType>(opInput.getType());
    inputShapes.push_back(inputType.getShape());
  }

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ConcatOp>::getOpRuntime, *this, inputShapes, inputs,
      getDim(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// TransposeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
TransposeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<TransposeOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDim0(), getDim1(), opConfig.outputLayout);
}

llvm::Expected<size_t>
TransposeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<TransposeOp>::getOpRuntime, *this, inputShape,
      inputs[0], getDim0(), getDim1(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// LinearOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LinearOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<LinearOp>::getOpConstraints, *this, deviceGrid,
      inputShapeA, inputs[0], inputShapeB, inputs[1], biasShape, biasLayout,
      opConfig.outputLayout, false, false);
}

llvm::Expected<size_t>
LinearOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<LinearOp>::getOpRuntime, *this, inputShapeA, inputs[0],
      inputShapeB, inputs[1], biasShape, biasLayout, opConfig.outputLayout,
      false, false);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MatmulOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<MatmulOp>::getOpConstraints, *this, deviceGrid,
      inputShapeA, inputs[0], inputShapeB, inputs[1], opConfig.outputLayout,
      false, false);
}

llvm::Expected<size_t>
MatmulOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<MatmulOp>::getOpRuntime, *this, inputShapeA, inputs[0],
      inputShapeB, inputs[1], opConfig.outputLayout, false, false);
}

//===----------------------------------------------------------------------===//
// Conv2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// If a config has been specified, use that. Otherwise, use the op property.
static Conv2dAttrs unpackConv2dAttrs(const OpConfig::OpSpecificAttrs &attrs,
                                     Conv2dOp op) {
  assert((std::holds_alternative<Conv2dAttrs>(attrs) ||
          std::holds_alternative<UninitializedAttrs>(attrs)) &&
         "Please create a Conv2dAttrs or leave it to be uninitialized.");

  if (std::holds_alternative<UninitializedAttrs>(attrs)) {
    return Conv2dAttrs{op.getConv2dConfig(), op.getComputeConfig()};
  }

  Conv2dAttrs conv2dAttrs = std::get<Conv2dAttrs>(attrs);

  return Conv2dAttrs{conv2dAttrs.conv2dConfig ? conv2dAttrs.conv2dConfig
                                              : op.getConv2dConfig(),
                     conv2dAttrs.deviceComputeKernelConfig
                         ? conv2dAttrs.deviceComputeKernelConfig
                         : op.getComputeConfig()};
}

llvm::Expected<op_model::OpConstraints>
Conv2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();
  Conv2dAttrs attr = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<Conv2dOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], weightShape, inputs[1], biasShape, biasLayout,
      getInChannels(), getOutChannels(), getBatchSize(), getInputHeight(),
      getInputWidth(), getKernelSize(), getStride(), getPadding(),
      getDilation(), getGroups(), attr.conv2dConfig,
      attr.deviceComputeKernelConfig, opConfig.outputLayout);
}

llvm::Expected<size_t>
Conv2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }
  Conv2dAttrs attr = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<Conv2dOp>::getOpRuntime, *this, inputShape, inputs[0],
      weightShape, inputs[1], biasShape, biasLayout, getInChannels(),
      getOutChannels(), getBatchSize(), getInputHeight(), getInputWidth(),
      getKernelSize(), getStride(), getPadding(), getDilation(), getGroups(),
      attr.conv2dConfig, attr.deviceComputeKernelConfig, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// If a config has been specified, use that. Otherwise, use the op property.
template <typename OpT>
static Conv2dAttrs unpackConv2dAttrs(const OpConfig::OpSpecificAttrs &attrs,
                                     OpT op) {
  assert((std::holds_alternative<Conv2dAttrs>(attrs) ||
          std::holds_alternative<UninitializedAttrs>(attrs)) &&
         "Please create a Conv2dAttrs or leave it to be uninitialized.");

  // ATM, ConvTranspose2dOp doesn't have a DeviceComputeKernelConfig attribute.
  // Default it to nullptr.
  if (std::holds_alternative<UninitializedAttrs>(attrs)) {
    return Conv2dAttrs{op.getConv2dConfig(), std::nullopt};
  }

  Conv2dAttrs conv2dAttrs = std::get<Conv2dAttrs>(attrs);

  return Conv2dAttrs{conv2dAttrs.conv2dConfig ? conv2dAttrs.conv2dConfig
                                              : op.getConv2dConfig(),
                     std::nullopt};
}

llvm::Expected<op_model::OpConstraints>
ConvTranspose2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                    const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  // If a conv config has been specified, use that. If not, read the op property
  Conv2dAttrs conv2dAttrs = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ConvTranspose2dOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], weightShape, inputs[1], biasShape, biasLayout,
      getInChannels(), getOutChannels(), getBatchSize(), getInputHeight(),
      getInputWidth(), getKernelSize(), getStride(), getPadding(),
      getOutputPadding(), getDilation(), getGroups(), conv2dAttrs.conv2dConfig,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ConvTranspose2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  // If a conv config has been specified, use that. If not, read the op property
  Conv2dAttrs conv2dAttrs = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ConvTranspose2dOp>::getOpRuntime, *this, inputShape,
      inputs[0], weightShape, inputs[1], biasShape, biasLayout, getInChannels(),
      getOutChannels(), getBatchSize(), getInputHeight(), getInputWidth(),
      getKernelSize(), getStride(), getPadding(), getOutputPadding(),
      getDilation(), getGroups(), conv2dAttrs.conv2dConfig,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// PrepareConv2dWeightsOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// query_op_runtime has to execute the op to measure the runtime (and not just
// invoke the op in NO_DISPATCH mode as query_op_constraint does). Therefore,
// any op that writes to memory, such as EmptyOp,ArangeOp, ZerosOp, OnesOp,
// etc., triggers a runtime error (`Writes are not supported during trace
// capture.`). As a consequence, we disable the runtime measurement for these
// ops. Alternatively, we could avoid defining the getOpRuntime API for such
// ops, but that would prevent us from the ultimate goal of supporting
// getOpRuntime and getOpConstraint for "all" ttnn ops. This is
// tracked/described here:
// https://github.com/tenstorrent/tt-mlir/issues/4199#issuecomment-3140045496
static llvm::Expected<size_t> issueErrorForGetOpRuntime(mlir::Operation *op) {
  auto opName = op->getName().getStringRef();
  return llvm::make_error<llvm::StringError>(
      "opRuntime is not supported for " + opName.str() +
          " since it requires memory IO.",
      llvm::inconvertibleErrorCode());
}

llvm::Expected<op_model::OpConstraints>
PrepareConv2dWeightsOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const ::llvm::ArrayRef<int64_t> weightShape =
      getWeightTensor().getType().getShape();
  Conv2dAttrs conv2dAttrs = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<PrepareConv2dWeightsOp>::getOpConstraints, *this,
      deviceGrid, inputs[0], weightShape, getInputMemoryConfig(),
      getInputTensorLayout(), getWeightsFormat(), getInChannels(),
      getOutChannels(), getBatchSize(), getInputHeight(), getInputWidth(),
      getKernelSize(), getStride(), getPadding(), getDilation(), getHasBias(),
      getGroups(), getInputDtype(), getOutputDtype(), conv2dAttrs.conv2dConfig,
      conv2dAttrs.deviceComputeKernelConfig, opConfig.outputLayout);
}

llvm::Expected<size_t>
PrepareConv2dWeightsOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                     const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
PrepareConv2dBiasOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                      const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const ::llvm::ArrayRef<int64_t> biasShape =
      getBiasTensor().getType().getShape();
  Conv2dAttrs conv2dAttrs = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<PrepareConv2dBiasOp>::getOpConstraints, *this,
      deviceGrid, inputs[0], biasShape, getInputMemoryConfig(),
      getInputTensorLayout(), getInChannels(), getOutChannels(), getBatchSize(),
      getInputHeight(), getInputWidth(), getKernelSize(), getStride(),
      getPadding(), getDilation(), getGroups(), getInputDtype(),
      getOutputDtype(), conv2dAttrs.conv2dConfig,
      conv2dAttrs.deviceComputeKernelConfig, opConfig.outputLayout);
}

llvm::Expected<size_t>
PrepareConv2dBiasOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                  const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MaxPool2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return detail::getPoolingOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MaxPool2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getPoolingOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// AvgPoo2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
AvgPool2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return detail::getPoolingOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
AvgPool2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getPoolingOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// BatchNormOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

struct BatchNormOptionalArgs {
  std::optional<llvm::ArrayRef<int64_t>> runningMeanShape = std::nullopt;
  std::optional<TTNNLayoutAttr> runningMeanLayout = std::nullopt;
  std::optional<llvm::ArrayRef<int64_t>> runningVarShape = std::nullopt;
  std::optional<TTNNLayoutAttr> runningVarLayout = std::nullopt;
  std::optional<llvm::ArrayRef<int64_t>> weightShape = std::nullopt;
  std::optional<TTNNLayoutAttr> weightLayout = std::nullopt;
  std::optional<llvm::ArrayRef<int64_t>> biasShape = std::nullopt;
  std::optional<TTNNLayoutAttr> biasLayout = std::nullopt;
};
static BatchNormOptionalArgs
unpackBatchNormOptionalArgs(const std::vector<TTNNLayoutAttr> &inputs,
                            BatchNormOp op) {
  BatchNormOptionalArgs ret;
  if (inputs.size() == 5) {
    ret.runningMeanShape = op.getRunningMean().getType().getShape();
    ret.runningVarShape = op.getRunningVar().getType().getShape();
    ret.weightShape = op.getWeight().getType().getShape();
    ret.biasShape = op.getBias().getType().getShape();
    ret.runningMeanLayout = inputs[1];
    ret.runningVarLayout = inputs[2];
    ret.weightLayout = inputs[3];
    ret.biasLayout = inputs[4];
  }
  return ret;
}

llvm::Expected<op_model::OpConstraints>
BatchNormOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert((inputs.size() == 1 || inputs.size() == 5) &&
         "ttnn::batch_norm can either have 1 input tensor (representing the "
         "main input) or 5 input tensors (representing main input tensor, "
         "running_mean, running_var, weight and bias). The usage of this op "
         "with 2-4 input tensors is discouraged as it's ambiguous.");

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const auto inputShape = getInput().getType().getShape();

  BatchNormOptionalArgs optionalArgs =
      unpackBatchNormOptionalArgs(inputs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<BatchNormOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], optionalArgs.runningMeanShape,
      optionalArgs.runningMeanLayout, optionalArgs.runningVarShape,
      optionalArgs.runningVarLayout, optionalArgs.weightShape,
      optionalArgs.weightLayout, optionalArgs.biasShape,
      optionalArgs.biasLayout, getEpsilon(), getTraining(), getMomentum(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
BatchNormOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert((inputs.size() == 1 || inputs.size() == 5) &&
         "ttnn::batch_norm can either have 1 input tensor (representing the "
         "main input) or 5 input tensors (representing main input tensor, "
         "running_mean, running_var, weight and bias). The usage of this op "
         "with 2-4 input tensors is discouraged as it's ambiguous.");

  const auto inputShape = getInput().getType().getShape();

  BatchNormOptionalArgs optionalArgs =
      unpackBatchNormOptionalArgs(inputs, *this);

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<BatchNormOp>::getOpRuntime, *this, inputShape,
      inputs[0], optionalArgs.runningMeanShape, optionalArgs.runningMeanLayout,
      optionalArgs.runningVarShape, optionalArgs.runningVarLayout,
      optionalArgs.weightShape, optionalArgs.weightLayout,
      optionalArgs.biasShape, optionalArgs.biasLayout, getEpsilon(),
      getTraining(), getMomentum(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ClampScalarOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ClampScalarOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ClampScalarOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getMin(), getMax(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ClampScalarOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ClampScalarOp>::getOpRuntime, *this, inputShape,
      inputs[0], getMin(), getMax(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// PermuteOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
PermuteOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<PermuteOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getPermutation(), getPadValue(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
PermuteOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<PermuteOp>::getOpRuntime, *this, inputShape, inputs[0],
      getPermutation(), getPadValue(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// UpsampleOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
UpsampleOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<UpsampleOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getScaleFactor(), getMode(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
UpsampleOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<UpsampleOp>::getOpRuntime, *this, inputShape, inputs[0],
      getScaleFactor(), getMode(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// EmbeddingOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
EmbeddingOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<EmbeddingOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], weightShape, inputs[1], opConfig.outputLayout);
}

llvm::Expected<size_t>
EmbeddingOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<EmbeddingOp>::getOpRuntime, *this, inputShape,
      inputs[0], weightShape, inputs[1], opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
EmbeddingBackwardOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                      const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  const auto inGradientShape = getInGradient().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<EmbeddingBackwardOp>::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], weightShape, inputs[1],
      inGradientShape, inputs[2], opConfig.outputLayout);
}

llvm::Expected<size_t>
EmbeddingBackwardOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                  const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  const auto inGradientShape = getInGradient().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<EmbeddingBackwardOp>::getOpRuntime, *this, inputShape,
      inputs[0], weightShape, inputs[1], inGradientShape, inputs[2],
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// EmptyOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
EmptyOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 0);

  const llvm::ArrayRef<int64_t> shape = getShape().getShape();
  const mlir::tt::ttcore::DataTypeAttr dtype = getDtypeAttr();
  const mlir::tt::ttnn::Layout layout = getLayoutAttr().getValue();
  const mlir::tt::ttnn::MemoryConfigAttr memoryConfig = getMemoryConfigAttr();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<mlir::tt::ttnn::EmptyOp>::getOpConstraints, *this,
      deviceGrid, shape, dtype, layout, memoryConfig, opConfig.outputLayout);
}

llvm::Expected<size_t>
EmptyOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

//===----------------------------------------------------------------------===//
// ArangeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ArangeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == 0);

  ::mlir::IntegerAttr startAttr = getStartAttr();
  ::mlir::IntegerAttr endAttr = getEndAttr();
  ::mlir::IntegerAttr stepAttr = getStepAttr();
  std::optional<mlir::tt::ttcore::DataType> dtype = getDtype();
  std::optional<mlir::tt::ttnn::MemoryConfigAttr> memConfig = getMemoryConfig();

  const mlir::tt::ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<mlir::tt::ttnn::ArangeOp>::getOpConstraints, *this,
      deviceGrid, startAttr, endAttr, stepAttr, dtype, memConfig,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ArangeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

//===----------------------------------------------------------------------===//
// ZerosOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ZerosOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getNamedFullOpConstraints(*this, inputs, opConfig);
}

// Similar to ArangeOp, we disable the runtime measurement for ZerosOp.
llvm::Expected<size_t>
ZerosOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

//===----------------------------------------------------------------------===//
// OnesOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
OnesOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getNamedFullOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
OnesOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

//===----------------------------------------------------------------------===//
// FullOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
FullOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 0);
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const mlir::tt::ttnn::ShapeAttr shape = getShape();
  const mlir::Attribute fillValue = getFillValue();
  const std::optional<mlir::tt::ttcore::DataType> dtype = getDtype();
  const std::optional<mlir::tt::ttnn::Layout> layout = getLayout();
  const std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      getMemoryConfig();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<mlir::tt::ttnn::FullOp>::getOpConstraints, *this,
      deviceGrid, shape, fillValue, dtype, layout, memoryConfig,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
FullOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(getOperation());
}

} // namespace mlir::tt::ttnn
