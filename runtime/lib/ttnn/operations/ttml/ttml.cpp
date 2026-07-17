// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ttml/ttml.h"
#include "operations/ttml/adamw.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn::operations::ttml {

// The one symbol the ttml library exports: dispatch a ttml op to its internal
// handler.
void run(const ::tt::target::ttnn::Operation *op, ProgramContext &context) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::AdamWOp:
    adamw(op->type_as_AdamWOp(), context);
    return;
  default:
    LOG_FATAL("Unsupported tt-train (ttml) op: ",
              ::tt::target::ttnn::EnumNameOpType(op->type_type()));
  }
}

} // namespace tt::runtime::ttnn::operations::ttml
