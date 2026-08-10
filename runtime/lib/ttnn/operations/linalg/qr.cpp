// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/linalg/qr.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::linalg::qr {
static void runLinalgQrOp(const ::tt::target::ttnn::QrOp *op,
                          ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->input());

  auto [q, r] = ::ttnn::qr(in, /*memory_config=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(0), q);
  tensorPool.insertTTNNTensorAndValidate(op->outputs()->Get(1), r);
}

void run(const ::tt::target::ttnn::QrOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runLinalgQrOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::linalg::qr
