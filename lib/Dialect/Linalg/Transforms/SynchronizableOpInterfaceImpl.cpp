// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/Linalg/Transforms/SynchronizableOpInterfaceImpl.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h"

using namespace mlir;
using namespace mlir::tt::d2m;

namespace {

struct LinalgGenericOpSynchronizableModel
    : public SynchronizableOpInterface::ExternalModel<
          LinalgGenericOpSynchronizableModel, linalg::GenericOp> {

  bool isProducer(Operation *op, OpOperand &operand) const {
    auto genericOp = cast<linalg::GenericOp>(op);
    // DPS init operands are producers (they write to the buffer)
    return genericOp.isDpsInit(&operand);
  }

  bool isConsumer(Operation *op, OpOperand &operand) const {
    auto genericOp = cast<linalg::GenericOp>(op);
    // DPS input operands are consumers (they read from the buffer)
    return genericOp.isDpsInput(&operand);
  }
};

} // namespace

void mlir::linalg::registerSynchronizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *) {
    linalg::GenericOp::attachInterface<LinalgGenericOpSynchronizableModel>(
        *ctx);
  });
}
