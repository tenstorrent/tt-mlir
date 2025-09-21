// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/tensor_serialization/dump_tensor.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::tensor_serialization {

void run(const ::tt::target::ttnn::DumpTensorOp *op, ProgramContext &context) {
  ::ttnn::Tensor input =
      context.getTensorPool().getTTNNTensorAndValidate(op->in());
  std::string filePath = op->file_path()->str();

  ::tt::tt_metal::dump_tensor_flatbuffer(filePath, input);
}

} // namespace tt::runtime::ttnn::operations::tensor_serialization
