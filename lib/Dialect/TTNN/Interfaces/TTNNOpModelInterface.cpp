// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/OpModel/TTNN/TTNNOpsModelCache.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace mlir::tt::ttnn {

namespace detail {

template <typename T>
inline std::string getAPITypeStr() {
  if constexpr (std::is_same_v<T, std::size_t>) {
    return "op_runtime";
  } else if constexpr (std::is_same_v<T, op_model::OpConstraints>) {
    return "op_constraint";
  }
}

enum class ReasonForLackOfSupport {
  NeedsMemoryIO,
  MissingMetalDefinition,
  NeedsMultiDevice,
  NoNeedForConstraintAPI,
  ArchitecturalMismatch,
};

inline std::string getReasonForLackOfSupportStr(ReasonForLackOfSupport reason) {
  switch (reason) {
  case ReasonForLackOfSupport::NeedsMemoryIO:
    return "needs memory IO";
  case ReasonForLackOfSupport::MissingMetalDefinition:
    return "missing metal definition";
  case ReasonForLackOfSupport::NeedsMultiDevice:
    return "needs multi-device";
  case ReasonForLackOfSupport::NoNeedForConstraintAPI:
    return "no need for constraint API";
  case ReasonForLackOfSupport::ArchitecturalMismatch:
    return "architectural mismatch between dialects";
  }
}

// This function issues a descriptive error message when the APIs are not
// supported for a specific reason.
template <typename T>
static llvm::Expected<T> issueError(mlir::Operation *op,
                                    ReasonForLackOfSupport reason) {
  static_assert(std::is_same_v<T, std::size_t> ||
                std::is_same_v<T, op_model::OpConstraints>);
  auto opName = op->getName().getStringRef();
  std::string error = getAPITypeStr<T>() + " is not supported for " +
                      opName.str() + ". Reason: [" +
                      getReasonForLackOfSupportStr(reason) + "]";
  return llvm::make_error<llvm::StringError>(error,
                                             llvm::inconvertibleErrorCode());
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Use case example:
//
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
static llvm::Expected<size_t>
issueErrorForGetOpRuntime(mlir::Operation *op, ReasonForLackOfSupport reason) {
  return issueError<std::size_t>(op, reason);
}
static llvm::Expected<op_model::OpConstraints>
issueErrorForGetOpConstraints(mlir::Operation *op,
                              ReasonForLackOfSupport reason) {
  return issueError<op_model::OpConstraints>(op, reason);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

template <typename OpT>
llvm::Expected<op_model::OpConstraints>
getQuantizationOpConstraints(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 3);
  const auto inputShape = op.getInput().getType().getShape();
  const auto scaleShape = op.getScale().getType().getShape();
  const auto zeroPointShape = op.getZeroPoint().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(op.getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(op.getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<OpT>::getOpConstraints, op, deviceGrid, inputShape,
      inputs[0], scaleShape, inputs[1], zeroPointShape, inputs[2], op.getAxis(),
      op.getOutputDtype(), opConfig.outputLayout);
}

template <typename OpT>
llvm::Expected<size_t>
getQuantizationOpRuntime(OpT op, const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 3);
  const auto inputShape = op.getInput().getType().getShape();
  const auto scaleShape = op.getScale().getType().getShape();
  const auto zeroPointShape = op.getZeroPoint().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<OpT>::getOpRuntime, op, inputShape, inputs[0],
      scaleShape, inputs[1], zeroPointShape, inputs[2], op.getAxis(),
      op.getOutputDtype(), opConfig.outputLayout);
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
// LogicalLeftShiftOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LogicalLeftShiftOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                     const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
LogicalLeftShiftOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
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
// BitwiseAndOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
BitwiseAndOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
BitwiseAndOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// BitwiseOrOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
BitwiseOrOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
BitwiseOrOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// BitwiseXorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
BitwiseXorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
BitwiseXorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// ScatterOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// (issue #4788) scatter is currently defined as a binary op in TTNNIR
// to be updated when it's fixed to match proper metal implementation

llvm::Expected<op_model::OpConstraints>
ScatterOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::ArchitecturalMismatch);
}

llvm::Expected<size_t>
ScatterOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::ArchitecturalMismatch);
}

//===----------------------------------------------------------------------===//
// Atan2Op - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
Atan2Op::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
Atan2Op::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// RemainderOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RemainderOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
RemainderOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return detail::getBinaryOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// PowOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
PowOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return detail::getBinaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
PowOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
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
// MaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getReductionOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  return getReductionOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// MinOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MinOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return getReductionOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
MinOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
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
      inputShape, inputs[0], getDimension(), getNumericStable(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<SoftmaxOp>::getOpRuntime, *this, inputShape, inputs[0],
      getDimension(), getNumericStable(), opConfig.outputLayout);
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
// SliceDynamicOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SliceDynamicOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                 const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShape = getInput().getType().getShape();
  const auto beginsShape = getBegins().getType().getShape();
  const auto endsShape = getEnds().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<SliceDynamicOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], beginsShape, inputs[1], endsShape, inputs[2],
      detail::convertOptionalArrayAttrToSmallVec(getStep()),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SliceDynamicOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShape = getInput().getType().getShape();
  const auto beginsShape = getBegins().getType().getShape();
  const auto endsShape = getEnds().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<SliceDynamicOp>::getOpRuntime, *this, inputShape,
      inputs[0], beginsShape, inputs[1], endsShape, inputs[2],
      detail::convertOptionalArrayAttrToSmallVec(getStep()),
      opConfig.outputLayout);
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
// GetDeviceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
GetDeviceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
GetDeviceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

//===----------------------------------------------------------------------===//
// FromDeviceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
FromDeviceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

llvm::Expected<size_t>
FromDeviceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

//===----------------------------------------------------------------------===//
// ToDeviceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ToDeviceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

llvm::Expected<size_t>
ToDeviceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}
//===----------------------------------------------------------------------===//
// ToDTypeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ToDTypeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
ToDTypeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
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
// MorehCumSumOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MorehCumSumOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<MorehCumSumOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDim(), opConfig.outputLayout);
}

llvm::Expected<size_t>
MorehCumSumOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<MorehCumSumOp>::getOpRuntime, *this, inputShape,
      inputs[0], getDim(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ConcatenateHeadsOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                     const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const auto inputShape = getInput().getType().getShape();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ConcatenateHeadsOp>::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], opConfig.outputLayout);
}

llvm::Expected<size_t>
ConcatenateHeadsOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                 const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ConcatenateHeadsOp>::getOpRuntime, *this, inputShape,
      inputs[0], opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

struct ScaledDotProductAttentionDecodeOptionalArgs {
  std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape = std::nullopt;
  std::optional<TTNNLayoutAttr> attentionMaskLayout = std::nullopt;
  std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape = std::nullopt;
  std::optional<TTNNLayoutAttr> attentionSinkLayout = std::nullopt;
};

static ScaledDotProductAttentionDecodeOptionalArgs
unpackScaledDotProductAttentionDecodeOptionalArgs(
    const std::vector<TTNNLayoutAttr> &inputs,
    ScaledDotProductAttentionDecodeOp op) {
  ScaledDotProductAttentionDecodeOptionalArgs ret;

  TypedValue<RankedTensorType> attentionMask = op.getAttentionMask();
  TypedValue<RankedTensorType> attentionSink = op.getAttentionSink();

  if (attentionMask && attentionSink) {
    ret.attentionMaskShape = attentionMask.getType().getShape();
    ret.attentionMaskLayout = inputs[4];
    ret.attentionSinkShape = attentionSink.getType().getShape();
    ret.attentionSinkLayout = inputs[5];
  } else if (attentionMask) {
    ret.attentionMaskShape = attentionMask.getType().getShape();
    ret.attentionMaskLayout = inputs[4];
  } else if (attentionSink) {
    ret.attentionSinkShape = attentionSink.getType().getShape();
    ret.attentionSinkLayout = inputs[4];
  } else {
    llvm_unreachable("All combinations of attention mask and attention sink "
                     "should have been handled");
  }

  return ret;
}

llvm::Expected<op_model::OpConstraints>
ScaledDotProductAttentionDecodeOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  // Clang tidy falsley determines that the underling float data in the
  // llvm::APFloat is freed more than once as APFloat is passed by value and
  // then destroyed at the end of this function.
  //
  // The compiler explorer session at the below link shows what occurs when an
  // optional value is set and passed by value in this manner
  // https://godbolt.org/z/sa9ojqqov
  //
  // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
  assert(inputs.size() >= 4 && inputs.size() <= 6 &&
         "ttnn::transformer::scaled_dot_product_attention_decode can have 4, "
         "5, or 6 "
         "input tensors");

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const auto queryShape = getQuery().getType().getShape();
  const auto keyShape = getKey().getType().getShape();
  const auto valueShape = getValue().getType().getShape();
  const auto curPosTensorShape = getCurPosTensor().getType().getShape();

  ScaledDotProductAttentionDecodeOptionalArgs optionalArgs =
      unpackScaledDotProductAttentionDecodeOptionalArgs(inputs, *this);

  auto scale = getScale();
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ScaledDotProductAttentionDecodeOp>::getOpConstraints,
      *this, deviceGrid, queryShape, inputs[0], keyShape, inputs[1], valueShape,
      inputs[2], curPosTensorShape, inputs[3], optionalArgs.attentionMaskShape,
      optionalArgs.attentionMaskLayout, optionalArgs.attentionSinkShape,
      optionalArgs.attentionSinkLayout, getIsCausal(), scale,
      opConfig.outputLayout);
  // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
}

llvm::Expected<size_t> ScaledDotProductAttentionDecodeOp::getOpRuntime(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  // See the comment in caledDotProductAttentionDecodeOp::getOpConstraints for
  // an explanation of this lint suppression.
  // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
  assert(inputs.size() >= 4 && inputs.size() <= 6 &&
         "ttnn::transformer::scaled_dot_product_attention_decode can have 4, "
         "5, or 6 "
         "input tensors");

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  const auto queryShape = getQuery().getType().getShape();
  const auto keyShape = getKey().getType().getShape();
  const auto valueShape = getValue().getType().getShape();
  const auto curPosTensorShape = getCurPosTensor().getType().getShape();

  ScaledDotProductAttentionDecodeOptionalArgs optionalArgs =
      unpackScaledDotProductAttentionDecodeOptionalArgs(inputs, *this);

  auto scale = getScale();
  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ScaledDotProductAttentionDecodeOp>::getOpRuntime, *this,
      queryShape, inputs[0], keyShape, inputs[1], valueShape, inputs[2],
      curPosTensorShape, inputs[3], optionalArgs.attentionMaskShape,
      optionalArgs.attentionMaskLayout, optionalArgs.attentionSinkShape,
      optionalArgs.attentionSinkLayout, getIsCausal(), scale,
      opConfig.outputLayout);
  // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ScaledDotProductAttentionOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  assert(inputs.size() >= 3 && inputs.size() <= 4 &&
         "ttnn::scaled_dot_product_attention_decode can have 3 or 5 operands "
         "input tensors");

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const auto queryShape = getQuery().getType().getShape();
  const auto keyShape = getKey().getType().getShape();
  const auto valueShape = getValue().getType().getShape();
  const std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape =
      getAttentionMask()
          ? std::make_optional(getAttentionMask().getType().getShape())
          : std::nullopt;
  const std::optional<TTNNLayoutAttr> attentionMaskLayout =
      getAttentionMask() ? std::make_optional(inputs[3]) : std::nullopt;
  bool isCausal = getIsCausal();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ScaledDotProductAttentionOp>::getOpConstraints, *this,
      deviceGrid, queryShape, inputs[0], keyShape, inputs[1], valueShape,
      inputs[2], attentionMaskShape, attentionMaskLayout, isCausal, getScale(),
      opConfig.outputLayout);
}

llvm::Expected<size_t> ScaledDotProductAttentionOp::getOpRuntime(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  assert(inputs.size() >= 3 && inputs.size() <= 4 &&
         "ttnn::scaled_dot_product_attention_decode can have 3 or 4 "
         "input tensors");

  assert(inputs.size() >= 3 && inputs.size() <= 4 &&
         "ttnn::scaled_dot_product_attention_decode can have 3 or 4 "
         "input tensors");

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  const auto queryShape = getQuery().getType().getShape();
  const auto keyShape = getKey().getType().getShape();
  const auto valueShape = getValue().getType().getShape();
  const std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape =
      getAttentionMask()
          ? std::make_optional(getAttentionMask().getType().getShape())
          : std::nullopt;
  const std::optional<TTNNLayoutAttr> attentionMaskLayout =
      getAttentionMask() ? std::make_optional(inputs[3]) : std::nullopt;
  bool isCausal = getIsCausal();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ScaledDotProductAttentionOp>::getOpRuntime, *this,
      queryShape, inputs[0], keyShape, inputs[1], valueShape, inputs[2],
      attentionMaskShape, attentionMaskLayout, isCausal, getScale(),
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOp - TTNN Op Model Interface
// ===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RotaryEmbeddingLlamaOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  assert(inputs.size() == 4);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();
  auto inputShape = getInput().getType().getShape();
  auto cosShape = getCosCache().getType().getShape();
  auto sinShape = getSinCache().getType().getShape();
  auto transMatShape = getTransMat().getType().getShape();
  bool isDecodeMode = getIsDecodeMode();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<RotaryEmbeddingLlamaOp>::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], cosShape, inputs[1], sinShape,
      inputs[2], transMatShape, inputs[3], isDecodeMode, opConfig.outputLayout);
}

llvm::Expected<size_t>
RotaryEmbeddingLlamaOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                     const OpConfig &opConfig) {
  assert(inputs.size() == 4);

  auto inputShape = getInput().getType().getShape();
  auto cosShape = getCosCache().getType().getShape();
  auto sinShape = getSinCache().getType().getShape();
  auto transMatShape = getTransMat().getType().getShape();
  bool isDecodeMode = getIsDecodeMode();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<RotaryEmbeddingLlamaOp>::getOpRuntime, *this,
      inputShape, inputs[0], cosShape, inputs[1], sinShape, inputs[2],
      transMatShape, inputs[3], isDecodeMode, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
NLPConcatHeadsOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                   const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();
  auto inputShape = getInput().getType().getShape();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<NLPConcatHeadsOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], opConfig.outputLayout);
}

llvm::Expected<size_t>
NLPConcatHeadsOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<NLPConcatHeadsOp>::getOpRuntime, *this, inputShape,
      inputs[0], opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//
llvm::Expected<op_model::OpConstraints>
NLPConcatHeadsDecodeOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const auto inputShape = getInput().getType().getShape();
  uint32_t numHeads = getNumHeads();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<NLPConcatHeadsDecodeOp>::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], numHeads, opConfig.outputLayout);
}

llvm::Expected<size_t>
NLPConcatHeadsDecodeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                     const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  uint32_t numHeads = getNumHeads();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<NLPConcatHeadsDecodeOp>::getOpRuntime, *this,
      inputShape, inputs[0], numHeads, opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RepeatInterleaveOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<RepeatInterleaveOp>::getOpConstraints, *this,
      deviceGrid, inputShape, inputs[0], getRepeats(), getDim(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
RepeatInterleaveOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                 const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<RepeatInterleaveOp>::getOpRuntime, *this, inputShape,
      inputs[0], getRepeats(), getDim(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// RepeatOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RepeatOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<RepeatOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getRepeatDims().getShape(), opConfig.outputLayout);
}

llvm::Expected<size_t>
RepeatOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<RepeatOp>::getOpRuntime, *this, inputShape, inputs[0],
      getRepeatDims().getShape(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// PadOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
PadOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<PadOp>::getOpConstraints, *this, deviceGrid, inputShape,
      inputs[0], getPadding(), getValue(), getUseMulticore(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
PadOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<PadOp>::getOpRuntime, *this, inputShape, inputs[0],
      getPadding(), getValue(), getUseMulticore(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// SortOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
SortOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<SortOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDim(), getDescending(), getStable(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
SortOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<SortOp>::getOpRuntime, *this, inputShape, inputs[0],
      getDim(), getDescending(), getStable(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ArgMaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ArgMaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<ArgMaxOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDim(), getKeepDim(), getUseMulticore(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
ArgMaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ArgMaxOp>::getOpRuntime, *this, inputShape, inputs[0],
      getDim(), getKeepDim(), getUseMulticore(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// ProdOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ProdOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
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
      op_model::OpModel<ProdOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getDimArg(), getKeepDim(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ProdOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

//===----------------------------------------------------------------------===//
// QuantizeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
QuantizeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return detail::getQuantizationOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
QuantizeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::getQuantizationOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// DequantizeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
DequantizeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return detail::getQuantizationOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
DequantizeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return detail::getQuantizationOpRuntime(*this, inputs, opConfig);
}

//===----------------------------------------------------------------------===//
// RequantizeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RequantizeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  assert(inputs.size() == 5);
  const auto inputShape = getInput().getType().getShape();
  const auto inScaleShape = getInScale().getType().getShape();
  const auto inZeroPointShape = getInZeroPoint().getType().getShape();
  const auto outScaleShape = getOutScale().getType().getShape();
  const auto outZeroPointShape = getOutZeroPoint().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<RequantizeOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], inScaleShape, inputs[1], inZeroPointShape,
      inputs[2], outScaleShape, inputs[3], outZeroPointShape, inputs[4],
      getAxis(), getOutputDtype(), opConfig.outputLayout);
}

llvm::Expected<size_t>
RequantizeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == 5);
  const auto inputShape = getInput().getType().getShape();
  const auto inScaleShape = getInScale().getType().getShape();
  const auto inZeroPointShape = getInZeroPoint().getType().getShape();
  const auto outScaleShape = getOutScale().getType().getShape();
  const auto outZeroPointShape = getOutZeroPoint().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<RequantizeOp>::getOpRuntime, *this, inputShape,
      inputs[0], inScaleShape, inputs[1], inZeroPointShape, inputs[2],
      outScaleShape, inputs[3], outZeroPointShape, inputs[4], getAxis(),
      getOutputDtype(), opConfig.outputLayout);
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
      opConfig.outputLayout, getTransposeA(), getTransposeB());
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
      getTransposeA(), getTransposeB());
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
      getTransposeA(), getTransposeB());
}

llvm::Expected<size_t>
MatmulOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const OpConfig &opConfig) {
  assert(inputs.size() == 2);

  const auto inputShapeA = getA().getType().getShape();
  const auto inputShapeB = getB().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<MatmulOp>::getOpRuntime, *this, inputShapeA, inputs[0],
      inputShapeB, inputs[1], opConfig.outputLayout, getTransposeA(),
      getTransposeB());
}

//===----------------------------------------------------------------------===//
// DeallocateOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
DeallocateOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  auto inputShape = getInput().getType().getShape();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<DeallocateOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getForce());
}

llvm::Expected<size_t>
DeallocateOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  assert(inputs.size() == 1);
  auto inputShape = getInput().getType().getShape();
  return opRuntimeCache().getOrCompute(
      op_model::OpModel<DeallocateOp>::getOpRuntime, *this, inputShape,
      inputs[0], getForce());
}

//===----------------------------------------------------------------------===//
// AllocOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
AllocOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
AllocOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// FillCacheOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
FillCacheOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  assert(inputs.size() == 2);
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  auto cacheShape = getCache().getType().getShape();
  auto inputShape = getInput().getType().getShape();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<FillCacheOp>::getOpConstraints, *this, deviceGrid,
      cacheShape, inputs[0], inputShape, inputs[1], getBatchOffset(),
      opConfig.outputLayout);
}

llvm::Expected<size_t>
FillCacheOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() == 2);
  auto cacheShape = getCache().getType().getShape();
  auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<FillCacheOp>::getOpRuntime, *this, cacheShape,
      inputs[0], inputShape, inputs[1], getBatchOffset(),
      opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
UpdateCacheOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  assert(inputs.size() == 3);
  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  auto cacheShape = getCache().getType().getShape();
  auto inputShape = getInput().getType().getShape();
  auto updateIndexShape = getUpdateIndex().getType().getShape();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<UpdateCacheOp>::getOpConstraints, *this, deviceGrid,
      cacheShape, inputs[0], inputShape, inputs[1], updateIndexShape, inputs[2],
      getBatchOffset(), opConfig.outputLayout);
}

llvm::Expected<size_t>
UpdateCacheOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 3);
  auto cacheShape = getCache().getType().getShape();
  auto inputShape = getInput().getType().getShape();
  auto updateIndexShape = getUpdateIndex().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<UpdateCacheOp>::getOpRuntime, *this, cacheShape,
      inputs[0], inputShape, inputs[1], updateIndexShape, inputs[2],
      getBatchOffset(), opConfig.outputLayout);
}

//===----------------------------------------------------------------------===//
// WriteTensorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
WriteTensorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
WriteTensorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
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
// RMSNormOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

struct RMSNormOptionalArgs {
  std::optional<llvm::ArrayRef<int64_t>> weightShape = std::nullopt;
  std::optional<TTNNLayoutAttr> weightLayout = std::nullopt;
  std::optional<llvm::ArrayRef<int64_t>> biasShape = std::nullopt;
  std::optional<TTNNLayoutAttr> biasLayout = std::nullopt;
};
static RMSNormOptionalArgs
unpackRMSNormOptionalArgs(const std::vector<TTNNLayoutAttr> &inputs,
                          RMSNormOp op) {
  RMSNormOptionalArgs ret;
  if (inputs.size() == 3) {
    ret.weightShape = op.getWeight().getType().getShape();
    ret.biasShape = op.getBias().getType().getShape();
    ret.weightLayout = inputs[1];
    ret.biasLayout = inputs[2];
  }
  return ret;
}

llvm::Expected<op_model::OpConstraints>
RMSNormOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert((inputs.size() == 1 || inputs.size() == 3) &&
         "ttnn::rms_norm can either have 1 input tensor (representing the "
         "main input) or 3 input tensors (representing main input tensor, "
         "weight and bias). The usage of this op with 2 input tensors is "
         "discouraged as it's ambiguous.");

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  const auto inputShape = getInput().getType().getShape();

  RMSNormOptionalArgs optionalArgs = unpackRMSNormOptionalArgs(inputs, *this);

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<RMSNormOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], optionalArgs.weightShape,
      optionalArgs.weightLayout, optionalArgs.biasShape,
      optionalArgs.biasLayout, getEpsilon(), opConfig.outputLayout);
}

llvm::Expected<size_t>
RMSNormOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  assert((inputs.size() == 1 || inputs.size() == 3) &&
         "ttnn::rms_norm can either have 1 input tensor (representing the "
         "main input) or 3 input tensors (representing main input tensor, "
         "weight and bias). The usage of this op with 2 input tensors is "
         "discouraged as it's ambiguous.");

  const auto inputShape = getInput().getType().getShape();

  RMSNormOptionalArgs optionalArgs = unpackRMSNormOptionalArgs(inputs, *this);

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<RMSNormOp>::getOpRuntime, *this, inputShape, inputs[0],
      optionalArgs.weightShape, optionalArgs.weightLayout,
      optionalArgs.biasShape, optionalArgs.biasLayout, getEpsilon(),
      opConfig.outputLayout);
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
// ClampTensorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ClampTensorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ClampTensorOp>::getOpConstraints, *this, deviceGrid,
      inputShape, inputs[0], getMin().getType().getShape(), inputs[1],
      getMax().getType().getShape(), inputs[2], opConfig.outputLayout);
}

llvm::Expected<size_t>
ClampTensorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  assert(inputs.size() == 3);

  const auto inputShape = getInput().getType().getShape();

  return opRuntimeCache().getOrCompute(
      op_model::OpModel<ClampTensorOp>::getOpRuntime, *this, inputShape,
      inputs[0], getMin().getType().getShape(), inputs[1],
      getMax().getType().getShape(), inputs[2], opConfig.outputLayout);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
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
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

//===----------------------------------------------------------------------===//
// AllGatherOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//
// AllGatherOp and ReduceScatterOp are not supported, since they have been
// removed from metal. See
// https://github.com/tenstorrent/tt-metal/commit/1ccd1c6480
llvm::Expected<op_model::OpConstraints>
AllGatherOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
AllGatherOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ReduceScatterOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                  const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
ReduceScatterOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// PointToPointOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// PointToPointOp is not supported rn. The reason is that the metal definition
// requires semaphore and multi-device, which is not available at the moment.
llvm::Expected<op_model::OpConstraints>
PointToPointOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                 const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMultiDevice);
}

llvm::Expected<size_t>
PointToPointOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMultiDevice);
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

// CollectivePermuteOp is not supported rn. The reason is that the metal
// definition is missing for this op.
llvm::Expected<op_model::OpConstraints>
CollectivePermuteOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                      const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}
llvm::Expected<size_t>
CollectivePermuteOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                  const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// AllReduceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
AllReduceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
AllReduceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// ConstantOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ConstantOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 0);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<ConstantOp>::getOpConstraints, *this, deviceGrid,
      getValue(), opConfig.outputLayout);
}

llvm::Expected<size_t>
ConstantOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

//===----------------------------------------------------------------------===//
// RandOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
RandOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  assert(inputs.size() == 0);

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }
  ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();

  return opConstraintsCache().getOrCompute(
      op_model::OpModel<mlir::tt::ttnn::RandOp>::getOpConstraints, *this,
      deviceGrid, getSize(), getDtype(), getMemoryConfig(), getLayout(),
      getLow(), getHigh(), getSeed(), opConfig.outputLayout);
}

llvm::Expected<size_t>
RandOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NeedsMemoryIO);
}

//===----------------------------------------------------------------------===//
// MeshShardOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
MeshShardOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
MeshShardOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// GenericOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
GenericOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
GenericOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// BeginTraceCaptureOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
BeginTraceCaptureOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                      const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
BeginTraceCaptureOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                  const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

//===----------------------------------------------------------------------===//
// EndTraceCaptureOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
EndTraceCaptureOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                    const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
EndTraceCaptureOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

//===----------------------------------------------------------------------===//
// ExecuteTraceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
ExecuteTraceOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                 const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
ExecuteTraceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

//===----------------------------------------------------------------------===//
// CaptureOrExecuteTraceOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
CaptureOrExecuteTraceOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

llvm::Expected<size_t>
CaptureOrExecuteTraceOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                                      const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::MissingMetalDefinition);
}

//===----------------------------------------------------------------------===//
// DumpTensorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
DumpTensorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
DumpTensorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

//===----------------------------------------------------------------------===//
// LoadTensorOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
LoadTensorOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  return issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

llvm::Expected<size_t>
LoadTensorOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                           const OpConfig &opConfig) {
  return issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::NoNeedForConstraintAPI);
}

} // namespace mlir::tt::ttnn
