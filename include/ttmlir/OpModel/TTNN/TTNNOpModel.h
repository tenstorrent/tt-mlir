// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <tuple>

namespace mlir::tt::op_model::ttnn {

struct ReluOpInterface {
  static bool isLegal(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
                      const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

  static std::tuple<size_t, size_t, size_t>
  getOpL1Usage(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
               const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
};

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
