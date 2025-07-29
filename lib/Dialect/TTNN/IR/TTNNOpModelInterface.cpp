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

#include "mlir/IR/Operation.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>

namespace mlir::tt::ttnn {

namespace detail {
llvm::Expected<bool> checkDeviceWorkerGrid(mlir::Operation *op) {
  auto deviceAttr = mlir::tt::ttcore::lookupDevice(op);
  assert(deviceAttr);
  return op_model::ttnn::Device::getDeviceConstraints(
      deviceAttr.getWorkerGrid());
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
llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<OpT>::getOpConstraints, op, deviceGrid,
      inputShape, inputs[0], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getUnaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                  const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<OpT>::getOpRuntime, op, inputShape, inputs[0],
      opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<OpT>::getOpConstraints, op, deviceGrid,
      inputShapeA, inputs[0], inputShapeB, inputs[1], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getBinaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                   const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = op.getLhs().getType().getShape();
  const auto inputShapeB = op.getRhs().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<OpT>::getOpRuntime, op, inputShapeA, inputs[0],
      inputShapeB, inputs[1], opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::ttnn::OpConstraints>
getTernaryOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShapeA = op.getFirst().getType().getShape();
  const auto inputShapeB = op.getSecond().getType().getShape();
  const auto inputShapeC = op.getThird().getType().getShape();
  const auto outputShape = op.getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::ttnn::OpModel<OpT>::getOpConstraints, op, deviceGrid,
      inputShapeA, inputs[0], inputShapeB, inputs[1], inputShapeC, inputs[2],
      outputShape, opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getTernaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShapeA = op.getFirst().getType().getShape();
  const auto inputShapeB = op.getSecond().getType().getShape();
  const auto inputShapeC = op.getThird().getType().getShape();
  const auto outputShape = op.getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<OpT>::getOpRuntime, op, inputShapeA, inputs[0],
      inputShapeB, inputs[1], inputShapeC, inputs[2], outputShape,
      opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<OpT>::getOpConstraints, op, deviceGrid,
      inputShape, inputs[0],
      detail::convertOptionalArrayAttrToSmallVec(op.getDimArg()),
      op.getKeepDim(), opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getReductionOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  const auto inputShape = op.getInput().getType().getShape();
  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<OpT>::getOpRuntime, op, inputShape, inputs[0],
      detail::convertOptionalArrayAttrToSmallVec(op.getDimArg()),
      op.getKeepDim(), opConfig.outputLayout);
}
} // namespace detail

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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
// CosOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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
// SigmoidOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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
// SubtractOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::SoftmaxOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getDimension(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::SoftmaxOp>::getOpRuntime, *this,
      inputShape, inputs[0], getDimension(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ReshapeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::ReshapeOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ReshapeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::ReshapeOp>::getOpRuntime, *this,
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SliceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SliceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::SliceOp>::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0],
      detail::convertArrayAttrToSmallVec(getBegins()),
      detail::convertArrayAttrToSmallVec(getEnds()),
      detail::convertArrayAttrToSmallVec(getStep()), opConfig.outputLayout);
}

llvm::Expected<size_t>
SliceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::SliceOp>::getOpRuntime, *this,
      inputShape, inputs[0], detail::convertArrayAttrToSmallVec(getBegins()),
      detail::convertArrayAttrToSmallVec(getEnds()),
      detail::convertArrayAttrToSmallVec(getStep()), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// TypecastOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::TypecastOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getDtypeAttr(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
TypecastOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::TypecastOp>::getOpRuntime, *this,
      inputShape, inputs[0], getDtypeAttr(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ToLayoutOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::ToLayoutOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getDtype(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ToLayoutOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  assert(opConfig.outputLayout.getLayout() == getLayout());

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::ToLayoutOp>::getOpRuntime, *this,
      inputShape, inputs[0], getDtype(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<
          mlir::tt::ttnn::ToMemoryConfigOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getMemoryConfig(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ToMemoryConfigOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::ToMemoryConfigOp>::getOpRuntime,
      *this, inputShape, inputs[0], getMemoryConfig(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ConcatOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::ConcatOp>::getOpConstraints,
      *this, deviceGrid, inputShapes, inputs, getDim(), opConfig.outputLayout);
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::ConcatOp>::getOpRuntime, *this,
      inputShapes, inputs, getDim(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// TransposeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::TransposeOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getDim0(), getDim1(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
TransposeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::TransposeOp>::getOpRuntime, *this,
      inputShape, inputs[0], getDim0(), getDim1(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// LinearOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
LinearOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout;

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
      op_model::ttnn::OpModel<mlir::tt::ttnn::LinearOp>::getOpConstraints,
      *this, deviceGrid, inputShapeA, inputs[0], inputShapeB, inputs[1],
      biasShape, biasLayout, opConfig.outputLayout, false, false);
}

llvm::Expected<size_t>
LinearOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::LinearOp>::getOpRuntime, *this,
      inputShapeA, inputs[0], inputShapeB, inputs[1], biasShape, biasLayout,
      opConfig.outputLayout, false, false);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::MatmulOp>::getOpConstraints,
      *this, deviceGrid, inputShapeA, inputs[0], inputShapeB, inputs[1],
      opConfig.outputLayout, false, false);
}

llvm::Expected<size_t>
MatmulOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::MatmulOp>::getOpRuntime, *this,
      inputShapeA, inputs[0], inputShapeB, inputs[1], opConfig.outputLayout,
      false, false);
}

//===----------------------------------------------------------------------===//
// Conv2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// If a config has been specified, use that. Otherwise, use the op property.
static Conv2dAttrs unpackConv2dAttrs(const OpConfig::OpSpecificAttrs &attrs,
                                     mlir::tt::ttnn::Conv2dOp op) {
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

llvm::Expected<op_model::ttnn::OpConstraints>
Conv2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout;

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
      op_model::ttnn::OpModel<mlir::tt::ttnn::Conv2dOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], weightShape, inputs[1],
      biasShape, biasLayout, getInChannels(), getOutChannels(), getBatchSize(),
      getInputHeight(), getInputWidth(), getKernelSize(), getStride(),
      getPadding(), getDilation(), getGroups(), attr.conv2dConfig,
      attr.deviceComputeKernelConfig, opConfig.outputLayout);
}

llvm::Expected<size_t>
Conv2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }
  Conv2dAttrs attr = unpackConv2dAttrs(opConfig.opSpecificAttrs, *this);

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::Conv2dOp>::getOpRuntime, *this,
      inputShape, inputs[0], weightShape, inputs[1], biasShape, biasLayout,
      getInChannels(), getOutChannels(), getBatchSize(), getInputHeight(),
      getInputWidth(), getKernelSize(), getStride(), getPadding(),
      getDilation(), getGroups(), attr.conv2dConfig,
      attr.deviceComputeKernelConfig, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// If a config has been specified, use that. Otherwise, use the op property.
static Conv2dAttrs
unpackConvTranspose2dAttrs(const OpConfig::OpSpecificAttrs &attrs,
                           mlir::tt::ttnn::ConvTranspose2dOp op) {
  assert((std::holds_alternative<Conv2dAttrs>(attrs) ||
          std::holds_alternative<UninitializedAttrs>(attrs)) &&
         "Please create a Conv2dAttrs or leave it to be uninitialized.");

  // ATM, ConvTranspose2dOp doesn't have a DeviceComputeKernelConfig attribute.
  // Default it to nullptr.
  if (std::holds_alternative<UninitializedAttrs>(attrs)) {
    return Conv2dAttrs{op.getConv2dConfig(), nullptr};
  }

  Conv2dAttrs conv2dAttrs = std::get<Conv2dAttrs>(attrs);

  return Conv2dAttrs{conv2dAttrs.conv2dConfig ? conv2dAttrs.conv2dConfig
                                              : op.getConv2dConfig(),
                     nullptr};
}

llvm::Expected<op_model::ttnn::OpConstraints>
ConvTranspose2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                    const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout;

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
  Conv2dAttrs conv2dAttrs =
      unpackConvTranspose2dAttrs(opConfig.opSpecificAttrs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::ttnn::OpModel<
          mlir::tt::ttnn::ConvTranspose2dOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], weightShape, inputs[1],
      biasShape, biasLayout, getInChannels(), getOutChannels(), getBatchSize(),
      getInputHeight(), getInputWidth(), getKernelSize(), getStride(),
      getPadding(), getOutputPadding(), getDilation(), getGroups(),
      conv2dAttrs.conv2dConfig, opConfig.outputLayout);
}

llvm::Expected<size_t>
ConvTranspose2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  assert(inputs.size() == (2 + (getBias() == nullptr ? 0 : 1)));

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout;

  if (inputs.size() == 3) {
    biasShape = getBias().getType().getShape();
    biasLayout = inputs[2];
  }

  // If a conv config has been specified, use that. If not, read the op property
  Conv2dAttrs conv2dAttrs =
      unpackConvTranspose2dAttrs(opConfig.opSpecificAttrs, *this);

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::ConvTranspose2dOp>::getOpRuntime,
      *this, inputShape, inputs[0], weightShape, inputs[1], biasShape,
      biasLayout, getInChannels(), getOutChannels(), getBatchSize(),
      getInputHeight(), getInputWidth(), getKernelSize(), getStride(),
      getPadding(), getOutputPadding(), getDilation(), getGroups(),
      conv2dAttrs.conv2dConfig, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
MaxPool2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::MaxPool2dOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getBatchSize(),
      getInputHeight(), getInputWidth(), getChannels(), getKernelSize(),
      getStride(), getPadding(), getDilation(), getCeilMode(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
MaxPool2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::MaxPool2dOp>::getOpRuntime, *this,
      inputShape, inputs[0], getBatchSize(), getInputHeight(), getInputWidth(),
      getChannels(), getKernelSize(), getStride(), getPadding(), getDilation(),
      getCeilMode(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ClampScalarOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::ClampScalarOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getMin(), getMax(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ClampScalarOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::ClampScalarOp>::getOpRuntime,
      *this, inputShape, inputs[0], getMin(), getMax(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// PermuteOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::PermuteOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getPermutation(), getPadValue(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
PermuteOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::PermuteOp>::getOpRuntime, *this,
      inputShape, inputs[0], getPermutation(), getPadValue(),
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// UpsampleOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::UpsampleOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], getScaleFactor(), getMode(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
UpsampleOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::UpsampleOp>::getOpRuntime, *this,
      inputShape, inputs[0], getScaleFactor(), getMode(),
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// EmbeddingOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
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
      op_model::ttnn::OpModel<mlir::tt::ttnn::EmbeddingOp>::getOpConstraints,
      *this, deviceGrid, inputShape, inputs[0], weightShape, inputs[1],
      opConfig.outputLayout);
}

llvm::Expected<size_t>
EmbeddingOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::EmbeddingOp>::getOpRuntime, *this,
      inputShape, inputs[0], weightShape, inputs[1], opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// EmptyOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
EmptyOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 0);
  assert(getDevice() && "Device is not set");

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
      op_model::ttnn::OpModel<mlir::tt::ttnn::EmptyOp>::getOpConstraints, *this,
      deviceGrid, shape, dtype, layout, memoryConfig);
}

llvm::Expected<size_t>
EmptyOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  assert(inputs.size() == 0);
  assert(getDevice() && "Device is not set");

  const llvm::ArrayRef<int64_t> shape = getShape().getShape();
  const mlir::tt::ttcore::DataTypeAttr dtype = getDtypeAttr();
  const mlir::tt::ttnn::Layout layout = getLayoutAttr().getValue();
  const mlir::tt::ttnn::MemoryConfigAttr memoryConfig = getMemoryConfigAttr();

  return opRuntimeCache().getOrCompute(
      op_model::ttnn::OpModel<mlir::tt::ttnn::EmptyOp>::getOpRuntime, *this,
      shape, dtype, layout, memoryConfig);
}

} // namespace mlir::tt::ttnn
