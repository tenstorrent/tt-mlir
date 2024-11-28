// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <vector>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PermuteOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  std::vector<int64_t> permutation(op->permutation()->begin(),
                                   op->permutation()->end());

  ::ttnn::Tensor out = ::ttnn::permute(in, permutation);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
  std::cout<<"input to permute logical shape: "<<in.get_logical_shape()<<std::endl;
  std::cout<<"output of permute logical shape: "<<out.get_logical_shape()<<std::endl;
}
} // namespace tt::runtime::ttnn::operations::data_movement
