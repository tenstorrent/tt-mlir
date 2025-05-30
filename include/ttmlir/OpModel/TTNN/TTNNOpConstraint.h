// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINT_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINT_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::op_model::ttnn {

/*
 * OpConstraints struct is used to store the constraints of an operation.
 * It is returned by the getOpConstraints method of the OpModel interface.
 * Note: The reason for seperating the definition of this struct from
 * TTNNOpModel.h is to avoid coupling TTNNOpModelInterface to all the code in
 * TTNNOpModel.h.
 */

struct OpConstraints {
  size_t cbL1PeakSize;                   // CB L1 peak allocation in bytes
  size_t tensorL1PeakSize;               // Tensor L1 peak allocation in bytes
  size_t outputL1BufferSize;             // Output L1 buffer allocation in bytes
  tt::ttnn::TTNNLayoutAttr outputLayout; // Layout of the output tensor
};

} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINT_H
