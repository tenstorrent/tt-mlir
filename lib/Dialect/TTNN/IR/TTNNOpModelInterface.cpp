// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/Operation.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir::tt::ttnn {

namespace detail {
llvm::Expected<bool> checkDeviceWorkerGrid(mlir::Operation *op) {
  auto deviceAttr = mlir::tt::lookupDevice(op);
  assert(deviceAttr);
  return op_model::ttnn::Device::getDeviceConstraints(
      deviceAttr.getWorkerGrid());
}

std::optional<llvm::SmallVector<int64_t>>
convertReductionArg(std::optional<mlir::ArrayAttr> arrayOpt) {
  if (!arrayOpt.has_value()) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t> reduceDims;

  for (const mlir::Attribute &reduceDim : *arrayOpt) {
    reduceDims.push_back(mlir::cast<mlir::IntegerAttr>(reduceDim).getInt());
  }

  return reduceDims;
}

} // namespace detail

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ReluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::ReluOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
ReluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {

  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  return op_model::ttnn::ReluOpInterface::getOpRuntime(
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SqrtOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
SqrtOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::SqrtOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
SqrtOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {

  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getType().getShape();

  return op_model::ttnn::SqrtOpInterface::getOpRuntime(
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
AddOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getLhs().getType().getShape();
  const auto inputShapeB = getRhs().getType().getShape();

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::AddOpInterface::getOpConstraints(
      deviceGrid, inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,

      opConfig.outputLayout);
}

llvm::Expected<size_t>
AddOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getLhs().getType().getShape();
  const auto inputShapeB = getRhs().getType().getShape();

  const auto outputShape = getType().getShape();

  return op_model::ttnn::AddOpInterface::getOpRuntime(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
SoftmaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::SoftmaxOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], getDimension(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  return op_model::ttnn::SoftmaxOpInterface::getOpRuntime(
      inputShape, inputs[0], getDimension(), outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// MeanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MeanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::MeanOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0],
      detail::convertReductionArg(getDimArg()), getKeepDim(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
MeanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return op_model::ttnn::MeanOpInterface::getOpRuntime(
      inputShape, inputs[0], detail::convertReductionArg(getDimArg()),
      getKeepDim(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ReshapeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ReshapeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::ReshapeOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

llvm::Expected<size_t>
ReshapeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return op_model::ttnn::ReshapeOpInterface::getOpRuntime(
      inputShape, inputs[0], outputShape, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// TypecastOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
TypecastOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::TypecastOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], getDtypeAttr(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
TypecastOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape = getResult().getType().getShape();

  return op_model::ttnn::TypecastOpInterface::getOpRuntime(
      inputShape, inputs[0], getDtypeAttr(), outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ToLayoutOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ToLayoutOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  assert(opConfig.outputLayout.getLayout() == getLayout());

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  auto deviceOperand = getDevice();
  const bool passDevicePtr = !deviceOperand;
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::ToLayoutOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], getDtype(), opConfig.outputLayout,
      passDevicePtr);
}

llvm::Expected<size_t>
ToLayoutOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  assert(opConfig.outputLayout.getLayout() == getLayout());

  const auto inputShape = getInput().getType().getShape();

  auto deviceOperand = getDevice();
  const bool passDevicePtr = !deviceOperand;

  return op_model::ttnn::ToLayoutOpInterface::getOpRuntime(
      inputShape, inputs[0], getDtype(), opConfig.outputLayout, passDevicePtr);
}

//===----------------------------------------------------------------------===//
// TransposeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
TransposeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::TransposeOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], getDim0(), getDim1(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
TransposeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return op_model::ttnn::TransposeOpInterface::getOpRuntime(
      inputShape, inputs[0], getDim0(), getDim1(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
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
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::MatmulOpInterface::getOpConstraints(
      deviceGrid, inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,
      opConfig.outputLayout, false, false);
}

llvm::Expected<size_t>
MatmulOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  const auto outputShape = getType().getShape();

  return op_model::ttnn::MatmulOpInterface::getOpRuntime(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,
      opConfig.outputLayout, false, false);
}

//===----------------------------------------------------------------------===//
// MultiplyOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MultiplyOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getLhs().getType().getShape();
  const auto inputShapeB = getRhs().getType().getShape();

  const auto outputShape = getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::MultiplyOpInterface::getOpConstraints(
      deviceGrid, inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,

      opConfig.outputLayout);
}

llvm::Expected<size_t>
MultiplyOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getLhs().getType().getShape();
  const auto inputShapeB = getRhs().getType().getShape();

  const auto outputShape = getType().getShape();

  return op_model::ttnn::MultiplyOpInterface::getOpRuntime(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// Conv2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
Conv2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == 2 || inputs.size() == 3);

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
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  // If a conv config has been specified, use that. If not, read the op property
  auto conv2dConfig =
      opConfig.config
          ? std::make_optional(mlir::cast<Conv2dConfigAttr>(opConfig.config))
          : getConv2dConfig();

  return op_model::ttnn::Conv2dOpInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], weightShape, inputs[1], biasShape,
      biasLayout, getInChannels(), getOutChannels(), getBatchSize(),
      getInputHeight(), getInputWidth(), getKernelSize(), getStride(),
      getPadding(), getDilation(), getGroups(), conv2dConfig, outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
Conv2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2 || inputs.size() == 3);

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
  auto conv2dConfig =
      opConfig.config
          ? std::make_optional(mlir::cast<Conv2dConfigAttr>(opConfig.config))
          : getConv2dConfig();

  return op_model::ttnn::Conv2dOpInterface::getOpRuntime(
      inputShape, inputs[0], weightShape, inputs[1], biasShape, biasLayout,
      getInChannels(), getOutChannels(), getBatchSize(), getInputHeight(),
      getInputWidth(), getKernelSize(), getStride(), getPadding(),
      getDilation(), getGroups(), conv2dConfig, outputShape,
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MaxPool2dOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  GridAttr deviceGrid = lookupDevice(getOperation()).getWorkerGrid();

  return op_model::ttnn::MaxPool2DInterface::getOpConstraints(
      deviceGrid, inputShape, inputs[0], getBatchSize(), getInputHeight(),
      getInputWidth(), getChannels(), getKernelSize(), getStride(),
      getPadding(), getDilation(), getCeilMode(), outputShape,
      opConfig.outputLayout);
}

llvm::Expected<size_t>
MaxPool2dOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  return op_model::ttnn::MaxPool2DInterface::getOpRuntime(
      inputShape, inputs[0], getBatchSize(), getInputHeight(), getInputWidth(),
      getChannels(), getKernelSize(), getStride(), getPadding(), getDilation(),
      getCeilMode(), outputShape, opConfig.outputLayout);
}

} // namespace mlir::tt::ttnn
