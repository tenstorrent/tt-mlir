// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include <mlir/IR/Operation.h>
namespace mlir::tt::ttir {

bool is_op_configuration_valid(const std::vector<Operation *> &producer_ops,
                               const std::vector<LayoutAttr> &producer_layouts,
                               Operation *consumer_Op,
                               const LayoutAttr &consumer_layout);

// If not operand data is provided, assume all operands have the same layout
// as the consumer op
bool is_op_configuration_valid(Operation *consumer_Op,
                               const LayoutAttr &consumer_layout);

}; // namespace mlir::tt::ttir
