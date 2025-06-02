// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H

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
  // ---------------------------------------------------------------------------
  // Parameterized constructor, should be used in most cases
  OpConstraints(size_t cbPeak, size_t tensorPeak, size_t outputBuffer,
                tt::ttnn::TTNNLayoutAttr layout)
      : cbL1PeakSize(cbPeak), tensorL1PeakSize(tensorPeak),
        outputL1BufferSize(outputBuffer), outputLayout(std::move(layout)) {}
  // ---------------------------------------------------------------------------
  // Default constructor, should be used only when the default value is intended
  // to be used, eg. when TTMLIR_ENABLE_OPMODEL is not defined.
  OpConstraints()
      : cbL1PeakSize(0), tensorL1PeakSize(0), outputL1BufferSize(0),
        outputLayout(nullptr) {}
};

} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
