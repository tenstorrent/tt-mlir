// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::op_model::ttnn {

struct ReluOpInterface {
  static bool isLegal(const mlir::tt::LayoutAttr &inputLayout,
                      const mlir::tt::LayoutAttr &outputLayout);

  static std::tuple<size_t, size_t, size_t>
  getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
               const mlir::tt::LayoutAttr &outputLayout);
};

} // namespace mlir::tt::op_model::ttnn
