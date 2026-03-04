// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::op_model {

/*
 * OpConstraints struct is used to store the constraints of an operation.
 * It is returned by the getOpConstraints method of the OpModel interface.
 * Note: The reason for separating the definition of this struct from
 * TTNNOpModel.h is to avoid coupling TTNNOpModelInterface to all the code in
 * TTNNOpModel.h.
 */

struct OpConstraints {
  size_t cbL1PeakSize;       // CB L1 peak allocation in bytes
  size_t tensorL1PeakSize;   // Tensor L1 peak allocation in bytes
  size_t peakL1MemorySize;   // Peak memory (CB+L1) allocation in bytes
  size_t outputL1BufferSize; // Output L1 buffer allocation in bytes
  llvm::SmallVector<tt::ttnn::TTNNLayoutAttr>
      outputLayouts; // Layouts of all output tensors (one layout per output
                     // tensor)
  // ---------------------------------------------------------------------------
  // Parameterized constructor, should be used in most cases
  OpConstraints(size_t cbPeak, size_t tensorPeak, size_t peakMemory,
                size_t outputBuffer,
                llvm::SmallVector<tt::ttnn::TTNNLayoutAttr> layouts)
      : cbL1PeakSize(cbPeak), tensorL1PeakSize(tensorPeak),
        peakL1MemorySize(peakMemory), outputL1BufferSize(outputBuffer),
        outputLayouts(std::move(layouts)) {}
  // ---------------------------------------------------------------------------
  // Default constructor, should be used only when the default value is intended
  // to be used, eg. when TTMLIR_ENABLE_OPMODEL is not defined.
  OpConstraints()
      : cbL1PeakSize(0), tensorL1PeakSize(0), peakL1MemorySize(0),
        outputL1BufferSize(0), outputLayouts({}) {}
};

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
