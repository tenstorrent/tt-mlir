// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#include "ttmlir/Dialect/TTNN/Utils/D2MOpCostModel.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Operation.h"

#include "llvm/Support/Error.h"

#include <cstddef>

namespace mlir::tt::ttnn {

namespace {

// D2M cost model: back-of-envelope L1 tensor estimates without calling the
// backend. We do not have kernel knowledge here, so we cannot compute
// cbL1PeakSize or the full peakL1MemorySize (see comments at construction).

static const char *const kNotSupportedMsg =
    "D2M cost model does not support this op";

uint64_t getL1SizeBytes(TTNNLayoutAttr layout) {
  if (!layout || !layout.hasL1BufferType()) {
    return 0;
  }
  return layout.getShardSizeInBytes();
}

bool isUnaryOp(Operation *op) {
  return llvm::isa<AbsOp, CbrtOp, CeilOp, SignOp, CosOp, ExpOp, ErfOp, ErfcOp,
                   FloorOp, GeluOp, IsFiniteOp, LogicalNotOp, BitwiseNotOp,
                   NegOp, TanOp, TanhOp, ReciprocalOp, ReluOp, SinOp, SqrtOp,
                   RsqrtOp, SigmoidOp, HardsigmoidOp, SiluOp, MishOp, LogOp,
                   Log1pOp, Expm1Op>(op);
}

bool isBinaryOp(Operation *op) {
  return llvm::isa<AddOp, DivideOp, MultiplyOp, SubtractOp, EqualOp, NotEqualOp,
                   GreaterEqualOp, GreaterThanOp, LessEqualOp, LessThanOp,
                   LogicalAndOp, LogicalOrOp, LogicalXorOp, LogicalRightShiftOp,
                   BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, MaximumOp,
                   MinimumOp, RemainderOp, LogicalLeftShiftOp, Atan2Op,
                   PowTensorOp>(op);
}

bool isReductionOp(Operation *op) {
  return llvm::isa<SumOp, MeanOp, MaxOp, MinOp>(op);
}

bool isMatmulOp(Operation *op) { return llvm::isa<MatmulOp, LinearOp>(op); }

bool isSupportedOp(Operation *op) {
  return isUnaryOp(op) || isBinaryOp(op) || isReductionOp(op) || isMatmulOp(op);
}

op_model::OpConstraints
estimateElementwiseConstraints(Operation *op,
                               const std::vector<TTNNLayoutAttr> &inputs,
                               const OpConfig &opConfig) {
  uint64_t outputL1 = utils::getOpOutputL1Usage(opConfig.outputLayout);
  uint64_t sumInputL1 = 0;
  for (TTNNLayoutAttr inputLayout : inputs) {
    sumInputL1 += getL1SizeBytes(inputLayout);
  }
  // Peak tensor L1 = output + all inputs (matches backend tensorL1PeakSize).
  // We cannot compute cbL1PeakSize or peakL1MemorySize here: both require
  // kernel circular-buffer usage, which only the backend knows.
  uint64_t tensorPeak = outputL1 + sumInputL1;
  return op_model::OpConstraints(/*cbL1PeakSize=*/0,
                                 /*tensorL1PeakSize=*/tensorPeak,
                                 /*peakL1MemorySize=*/tensorPeak,
                                 /*outputL1BufferSize=*/outputL1,
                                 opConfig.outputLayout);
}

op_model::OpConstraints
estimateReductionConstraints(Operation *op,
                             const std::vector<TTNNLayoutAttr> &inputs,
                             const OpConfig &opConfig) {
  assert(inputs.size() == 1 && "reduction has one input");
  uint64_t outputL1 = utils::getOpOutputL1Usage(opConfig.outputLayout);
  uint64_t inputL1 = getL1SizeBytes(inputs[0]);
  uint64_t peak = outputL1 + inputL1;
  // cbL1PeakSize and peakL1MemorySize need kernel CB usage; we use 0 / tensor.
  return op_model::OpConstraints(/*cbL1PeakSize=*/0,
                                 /*tensorL1PeakSize=*/peak,
                                 /*peakL1MemorySize=*/peak,
                                 /*outputL1BufferSize=*/outputL1,
                                 opConfig.outputLayout);
}

op_model::OpConstraints
estimateMatmulConstraints(Operation *op,
                          const std::vector<TTNNLayoutAttr> &inputs,
                          const OpConfig &opConfig) {
  assert(inputs.size() >= 2 && "matmul/linear has at least two inputs");
  uint64_t outputL1 = utils::getOpOutputL1Usage(opConfig.outputLayout);
  uint64_t inputAL1 = getL1SizeBytes(inputs[0]);
  uint64_t inputBL1 = getL1SizeBytes(inputs[1]);
  uint64_t peak = outputL1 + inputAL1 + inputBL1;
  // cbL1PeakSize and peakL1MemorySize need kernel CB usage; we use 0 / tensor.
  return op_model::OpConstraints(/*cbL1PeakSize=*/0,
                                 /*tensorL1PeakSize=*/peak,
                                 /*peakL1MemorySize=*/peak,
                                 /*outputL1BufferSize=*/outputL1,
                                 opConfig.outputLayout);
}

} // namespace

llvm::Expected<op_model::OpConstraints>
estimateOpConstraints(Operation *op, const std::vector<TTNNLayoutAttr> &inputs,
                      const OpConfig &opConfig) {
  if (!op || !isSupportedOp(op)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   kNotSupportedMsg);
  }
  if (isUnaryOp(op) || isBinaryOp(op)) {
    return estimateElementwiseConstraints(op, inputs, opConfig);
  }
  if (isReductionOp(op)) {
    return estimateReductionConstraints(op, inputs, opConfig);
  }
  if (isMatmulOp(op)) {
    return estimateMatmulConstraints(op, inputs, opConfig);
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kNotSupportedMsg);
}

llvm::Expected<size_t>
estimateOpRuntime(Operation *op, const std::vector<TTNNLayoutAttr> &inputs,
                  const OpConfig &opConfig) {
  if (!op || !isSupportedOp(op)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   kNotSupportedMsg);
  }
  (void)inputs;
  (void)opConfig;
  return static_cast<size_t>(0);
}

} // namespace mlir::tt::ttnn
