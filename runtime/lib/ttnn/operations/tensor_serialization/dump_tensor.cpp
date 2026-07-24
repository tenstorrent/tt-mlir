// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/tensor_serialization/dump_tensor.h"
#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::tensor_serialization {

void run(const ::tt::target::ttnn::DumpTensorOp *op, ProgramContext &context) {
  const ::ttnn::Tensor &input =
      context.getTensorPool().getTTNNTensorAndValidate(op->in());
  std::string filePath = op->file_path()->str();

  ::ttnn::dump_tensor_flatbuffer(filePath, input);
}

} // namespace tt::runtime::ttnn::operations::tensor_serialization
