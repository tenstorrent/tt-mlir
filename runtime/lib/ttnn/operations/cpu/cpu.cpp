// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/utils.h"

namespace tt::runtime::ttnn::operations::cpu {

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  common::WrappedFunc fn = context.getDylibManager().getFunc(
      op->dylib_id(), op->func_name()->c_str());
  LOG_ASSERT(fn != nullptr);

  const auto *fbInputs = op->ins();

  std::vector<std::vector<int64_t>> allSizesAndStrides;

  std::function<void *(const tt::target::ttnn::TensorRef *)> getTensorDataPtr =
      [&context](const tt::target::ttnn::TensorRef *ref) -> void * {
    const auto &tens = context.getTensorPool().getTTNNTensorAndValidate(ref);
    return ::tt::runtime::ttnn::utils::getRawHostDataPtr(tens);
  };

  auto dylibInputs = tt::runtime::common::packTensors(
      fbInputs, getTensorDataPtr, allSizesAndStrides);

  // Call the CPU function and get returned outputs.
  common::WrappedTensor *outputArray = fn(dylibInputs.data());

  // Callback for tensor creation from WrappedTensor.
  common::CreateTensorCallbackType<::ttnn::Tensor,
                                   ::tt::target::ttnn::TensorRef>
      createTensor = [](const tt::target::ttnn::TensorRef *ref,
                        std::shared_ptr<void> dataPtr) -> ::ttnn::Tensor {
    ::ttnn::Shape shape = utils::toTTNNShape(*ref->desc()->shape());
    ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
        ref->desc()->layout()->memory_desc()->data_type());

    ::ttnn::Tensor tensor;
    switch (dtype) {
    case ::ttnn::DataType::FLOAT32:
      tensor = ::tt::runtime::ttnn::utils::createBorrowedTTNNTensor<float>(
          dataPtr, shape);
      break;
    case ::ttnn::DataType::INT32:
      tensor = ::tt::runtime::ttnn::utils::createBorrowedTTNNTensor<int32_t>(
          dataPtr, shape);
      break;
    default:
      LOG_FATAL("Unsupported data type for CPU op output");
    }

    return tensor;
  };

  // Unpack outputs and insert into tensor pool.
  auto outputs = common::unpackTensors<::ttnn::Tensor>(
      outputArray, op->outs()->size(), op->outs(), createTensor);

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertTTNNTensorAndValidate(op->outs()->Get(i),
                                                        outputs[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::cpu
