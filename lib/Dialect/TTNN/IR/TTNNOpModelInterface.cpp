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
#include <llvm/ADT/Hashing.h>
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

} // namespace detail

using BinaryOpConstraintsFunc =
    std::function<llvm::Expected<op_model::ttnn::OpConstraints>(
        ttcore::GridAttr, llvm::ArrayRef<int64_t>,
        mlir::tt::ttnn::TTNNLayoutAttr, llvm::ArrayRef<int64_t>,
        mlir::tt::ttnn::TTNNLayoutAttr, llvm::ArrayRef<int64_t>,
        mlir::tt::ttnn::TTNNLayoutAttr)>;

using BinaryOpRuntimeFunc = std::function<llvm::Expected<size_t>(
    llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
    llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
    llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr)>;

template <typename OpT>
llvm::Expected<op_model::ttnn::OpConstraints>
getBinaryOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig,
                       BinaryOpConstraintsFunc getConstraintsFunc) {
  assert(inputs.size() == 2);

  const auto inputShapeA = op.getLhs().getType().getShape();
  const auto inputShapeB = op.getRhs().getType().getShape();
  const auto outputShape = op.getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      getConstraintsFunc, op, deviceGrid, inputShapeA, inputs[0], inputShapeB,
      inputs[1], outputShape, opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getBinaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                   const OpConfig &opConfig,
                   BinaryOpRuntimeFunc getRuntimeFunc) {
  assert(inputs.size() == 2);

  const auto inputShapeA = op.getLhs().getType().getShape();
  const auto inputShapeB = op.getRhs().getType().getShape();
  const auto outputShape = op.getType().getShape();

  return opRuntimeCache.getOrCompute(getRuntimeFunc, op, inputShapeA, inputs[0],
                                     inputShapeB, inputs[1], outputShape,
                                     opConfig.outputLayout);
}

using ReductionOpConstraintsFunc =
    std::function<llvm::Expected<op_model::ttnn::OpConstraints>(
        ttcore::GridAttr, llvm::ArrayRef<int64_t>,
        mlir::tt::ttnn::TTNNLayoutAttr, std::optional<llvm::ArrayRef<int64_t>>,
        bool, mlir::tt::ttnn::TTNNLayoutAttr)>;

using ReductionOpRuntimeFunc = std::function<llvm::Expected<size_t>(
    llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
    std::optional<llvm::ArrayRef<int64_t>>, bool,
    mlir::tt::ttnn::TTNNLayoutAttr)>;

template <typename OpT>
llvm::Expected<op_model::ttnn::OpConstraints>
getReductionOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig,
                          ReductionOpConstraintsFunc getConstraintsFunc) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      getConstraintsFunc, op, deviceGrid, inputShape, inputs[0],
      detail::convertOptionalArrayAttrToSmallVec(op.getDimArg()),
      op.getKeepDim(), opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getReductionOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig,
                      ReductionOpRuntimeFunc getRuntimeFunc) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();

  return opRuntimeCache.getOrCompute(
      getRuntimeFunc, op, inputShape, inputs[0],
      detail::convertOptionalArrayAttrToSmallVec(op.getDimArg()),
      op.getKeepDim(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
using UnaryOpConstraintsFunc =
    std::function<llvm::Expected<op_model::ttnn::OpConstraints>(
        ttcore::GridAttr, llvm::ArrayRef<int64_t>,
        mlir::tt::ttnn::TTNNLayoutAttr, llvm::ArrayRef<int64_t>,
        mlir::tt::ttnn::TTNNLayoutAttr)>;

using UnaryOpRuntimeFunc = std::function<llvm::Expected<size_t>(
    llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
    llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr)>;

template <typename OpT>
llvm::Expected<op_model::ttnn::OpConstraints>
getUnaryOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig,
                      UnaryOpConstraintsFunc getConstraintsFunc) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();
  const auto outputShape = op.getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();
  return opConstraintsCache.getOrCompute(getConstraintsFunc, op, deviceGrid,
                                         inputShape, inputs[0], outputShape,
                                         opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getUnaryOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                  const OpConfig &opConfig, UnaryOpRuntimeFunc getRuntimeFunc) {
  assert(inputs.size() == 1);

  const auto inputShape = op.getInput().getType().getShape();
  const auto outputShape = op.getType().getShape();

  return opRuntimeCache.getOrCompute(getRuntimeFunc, op, inputShape, inputs[0],
                                     outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
ReluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return getUnaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::ReluOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
ReluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {

  return getUnaryOpRuntime(*this, inputs, opConfig,
                           op_model::ttnn::ReluOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// SinOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SinOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getUnaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::SinOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
SinOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {

  return getUnaryOpRuntime(*this, inputs, opConfig,
                           op_model::ttnn::SinOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// CosOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
CosOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getUnaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::CosOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
CosOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {

  return getUnaryOpRuntime(*this, inputs, opConfig,
                           op_model::ttnn::CosOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// ReciprocalOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
ReciprocalOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return getUnaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::ReciprocalOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
ReciprocalOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {

  return getUnaryOpRuntime(*this, inputs, opConfig,
                           op_model::ttnn::ReciprocalOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// SqrtOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SqrtOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::SqrtOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
SqrtOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {

  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::SqrtOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SigmoidOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SigmoidOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::SigmoidOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
SigmoidOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {

  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::SigmoidOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
AddOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getBinaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::AddOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
AddOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return getBinaryOpRuntime(*this, inputs, opConfig,
                            op_model::ttnn::AddOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SoftmaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
  return opConstraintsCache.getOrCompute(
      op_model::ttnn::SoftmaxOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDimension(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::SoftmaxOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getDimension(), outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// MeanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
MeanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return getReductionOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::MeanOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
MeanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return getReductionOpRuntime(*this, inputs, opConfig,
                               op_model::ttnn::MeanOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// SumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getReductionOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::SumOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
SumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return getReductionOpRuntime(*this, inputs, opConfig,
                               op_model::ttnn::SumOpInterface::getOpRuntime);
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::ReshapeOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
ReshapeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::ReshapeOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SliceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SliceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::SliceOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], detail::convertArrayAttrToSmallVec(getBegins()),
      detail::convertArrayAttrToSmallVec(getEnds()),
      detail::convertArrayAttrToSmallVec(getStep()), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SliceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::SliceOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], detail::convertArrayAttrToSmallVec(getBegins()),
      detail::convertArrayAttrToSmallVec(getEnds()),
      detail::convertArrayAttrToSmallVec(getStep()), outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// TypecastOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
TypecastOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::TypecastOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDtypeAttr(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
TypecastOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::TypecastOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getDtypeAttr(), outputShape, opConfig.outputLayout);
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::ToLayoutOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDtype(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ToLayoutOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  assert(opConfig.outputLayout.getLayout() == getLayout());

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::ToLayoutOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getDtype(), opConfig.outputLayout);
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::ConcatOpInterface::getOpConstraints, *this, deviceGrid,
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

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::ConcatOpInterface::getOpRuntime, *this, inputShapes,
      inputs, getDim(), opConfig.outputLayout);
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::TransposeOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDim0(), getDim1(), opConfig.outputLayout);
}

llvm::Expected<size_t>
TransposeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::TransposeOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getDim0(), getDim1(), opConfig.outputLayout);
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

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::MatmulOpInterface::getOpConstraints, *this, deviceGrid,
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,
      opConfig.outputLayout, false, false);
}

llvm::Expected<size_t>
MatmulOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  const auto outputShape = getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::MatmulOpInterface::getOpRuntime, *this, inputShapeA,
      inputs[0], inputShapeB, inputs[1], outputShape, opConfig.outputLayout,
      false, false);
}

//===----------------------------------------------------------------------===//
// MultiplyOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
MultiplyOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return getBinaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::MultiplyOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
MultiplyOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return getBinaryOpRuntime(*this, inputs, opConfig,
                            op_model::ttnn::MultiplyOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// Conv2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

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

  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  // If a conv config has been specified, use that. If not, read the op property
  std::optional<Conv2dConfigAttr> conv2dConfig = std::nullopt;
  if (opConfig.opSpecificAttr) {
    assert(mlir::isa<Conv2dConfigAttr>(opConfig.opSpecificAttr) &&
           "Unexpected type for OpConfig.opSpecificAttr. Expected "
           "Conv2ConfigAttr");
    conv2dConfig = mlir::cast<Conv2dConfigAttr>(opConfig.opSpecificAttr);
  } else {
    conv2dConfig = getConv2dConfig();
  }

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::Conv2dOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], weightShape, inputs[1], biasShape, biasLayout,
      getInChannels(), getOutChannels(), getBatchSize(), getInputHeight(),
      getInputWidth(), getKernelSize(), getStride(), getPadding(),
      getDilation(), getGroups(), conv2dConfig, outputShape,
      opConfig.outputLayout);
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

  const auto outputShape = getResult().getType().getShape();

  // If a conv config has been specified, use that. If not, read the op property
  std::optional<Conv2dConfigAttr> conv2dConfig = std::nullopt;
  if (opConfig.opSpecificAttr) {
    assert(mlir::isa<Conv2dConfigAttr>(opConfig.opSpecificAttr) &&
           "Unexpected type for OpConfig.confopSpecificAttrig. Expected "
           "Conv2ConfigAttr");
    conv2dConfig = mlir::cast<Conv2dConfigAttr>(opConfig.opSpecificAttr);
  } else {
    conv2dConfig = getConv2dConfig();
  }

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::Conv2dOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], weightShape, inputs[1], biasShape, biasLayout, getInChannels(),
      getOutChannels(), getBatchSize(), getInputHeight(), getInputWidth(),
      getKernelSize(), getStride(), getPadding(), getDilation(), getGroups(),
      conv2dConfig, outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

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

  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  // If a conv config has been specified, use that. If not, read the op property
  std::optional<Conv2dConfigAttr> conv2dConfig = std::nullopt;
  if (opConfig.opSpecificAttr) {
    assert(mlir::isa<Conv2dConfigAttr>(opConfig.opSpecificAttr) &&
           "Unexpected type for OpConfig.opSpecificAttr. Expected "
           "Conv2ConfigAttr");
    conv2dConfig = mlir::cast<Conv2dConfigAttr>(opConfig.opSpecificAttr);
  } else {
    conv2dConfig = getConv2dConfig();
  }

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::ConvTranspose2dOpInterface::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], weightShape, inputs[1], biasShape,
      biasLayout, getInChannels(), getOutChannels(), getBatchSize(),
      getInputHeight(), getInputWidth(), getKernelSize(), getStride(),
      getPadding(), getOutputPadding(), getDilation(), getGroups(),
      conv2dConfig, outputShape, opConfig.outputLayout);
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

  const auto outputShape = getResult().getType().getShape();

  // If a conv config has been specified, use that. If not, read the op property
  std::optional<Conv2dConfigAttr> conv2dConfig = std::nullopt;
  if (opConfig.opSpecificAttr) {
    assert(mlir::isa<Conv2dConfigAttr>(opConfig.opSpecificAttr) &&
           "Unexpected type for OpConfig.confopSpecificAttrig. Expected "
           "Conv2ConfigAttr");
    conv2dConfig = mlir::cast<Conv2dConfigAttr>(opConfig.opSpecificAttr);
  } else {
    conv2dConfig = getConv2dConfig();
  }

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::ConvTranspose2dOpInterface::getOpRuntime, *this,
      inputShape, inputs[0], weightShape, inputs[1], biasShape, biasLayout,
      getInChannels(), getOutChannels(), getBatchSize(), getInputHeight(),
      getInputWidth(), getKernelSize(), getStride(), getPadding(),
      getOutputPadding(), getDilation(), getGroups(), conv2dConfig, outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
MaxPool2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::MaxPool2DOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getBatchSize(), getInputHeight(), getInputWidth(),
      getChannels(), getKernelSize(), getStride(), getPadding(), getDilation(),
      getCeilMode(), outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
MaxPool2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::MaxPool2DOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getBatchSize(), getInputHeight(), getInputWidth(),
      getChannels(), getKernelSize(), getStride(), getPadding(), getDilation(),
      getCeilMode(), outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ClampScalarOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
ClampScalarOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::ClampScalarOpInterface::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], getMin(), getMax(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ClampScalarOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::ClampScalarOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getMin(), getMax(), outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// PermuteOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
PermuteOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::PermuteOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getPermutation(), getPadValue(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
PermuteOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::PermuteOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getPermutation(), getPadValue(), outputShape,
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

  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::UpsampleOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getScaleFactor(), getMode(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
UpsampleOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::UpsampleOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], getScaleFactor(), getMode(), outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SubtractOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
SubtractOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return getBinaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::SubtractOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
SubtractOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return getBinaryOpRuntime(*this, inputs, opConfig,
                            op_model::ttnn::SubtractOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// MaximumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
MaximumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return getBinaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::MaximumOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
MaximumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getBinaryOpRuntime(*this, inputs, opConfig,
                            op_model::ttnn::MaximumOpInterface::getOpRuntime);
}

//===----------------------------------------------------------------------===//
// MinimumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::ttnn::OpConstraints>
MinimumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return getBinaryOpConstraints(
      *this, inputs, opConfig,
      op_model::ttnn::MinimumOpInterface::getOpConstraints);
}

llvm::Expected<size_t>
MinimumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getBinaryOpRuntime(*this, inputs, opConfig,
                            op_model::ttnn::MinimumOpInterface::getOpRuntime);
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
  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache.getOrCompute(
      op_model::ttnn::EmbeddingOpInterface::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], weightShape, inputs[1], outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
EmbeddingOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShape = getInput().getType().getShape();
  const auto weightShape = getWeight().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return opRuntimeCache.getOrCompute(
      op_model::ttnn::EmbeddingOpInterface::getOpRuntime, *this, inputShape,
      inputs[0], weightShape, inputs[1], outputShape, opConfig.outputLayout);
}

} // namespace mlir::tt::ttnn
