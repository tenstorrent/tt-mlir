// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/creation/empty.h"
#include "operations/creation/full.h"
#include "operations/data_movement/concat.h"
#include "operations/data_movement/reshape.h"
#include "operations/data_movement/transpose.h"
#include "operations/deletion/dealloc.h"
#include "operations/eltwise/binary.h"
#include "operations/eltwise/unary.h"
#include "operations/embedding/embedding.h"
#include "operations/layout/from_device.h"
#include "operations/layout/to_device.h"
#include "operations/layout/to_layout.h"
#include "operations/layout/to_memory_config.h"
#include "operations/matmul/matmul.h"
#include "operations/normalization/softmax.h"
#include "operations/pool/maxpool2d.h"
#include "operations/reduction/reduction.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn {

struct ProgramExecutor {
  ProgramContext context;
  ProgramExecutor(const TensorMap &liveTensors, const DeviceMap &allDevices)
      : context(ProgramContext(liveTensors, allDevices)) {}

  void execute(const ::tt::target::ttnn::Program *program) {
    for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
      runOperation(op);
    }
  }

private:
  void runOperation(const ::tt::target::ttnn::Operation *op);
};

void ProgramExecutor::runOperation(const ::tt::target::ttnn::Operation *op) {

  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    std::cout << "OpType: GetDeviceOp" << std::endl;
    operations::context::run(op->type_as_GetDeviceOp(), context);
    std::cout << "Op finished: GetDeviceOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    std::cout << "OpType: ToMemoryConfigOp" << std::endl;
    operations::layout::run(op->type_as_ToMemoryConfigOp(), context);
    std::cout << "Op finished: ToMemoryConfigOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    std::cout << "OpType: ToLayoutOp" << std::endl;
    operations::layout::run(op->type_as_ToLayoutOp(), context);
    std::cout << "Op finished: ToLayoutOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    std::cout << "OpType: ToDeviceOp" << std::endl;
    operations::layout::run(op->type_as_ToDeviceOp(), context);
    std::cout << "Op finished: ToDeviceOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    return operations::layout::run(op->type_as_FromDeviceOp(), context);
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    std::cout << "OpType: EmptyOp" << std::endl;
    operations::creation::run(op->type_as_EmptyOp(), context);
    std::cout << "Op finished: EmptyOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    std::cout << "OpType: FullOp" << std::endl;
    operations::creation::run(op->type_as_FullOp(), context);
    std::cout << "Op finished: FullOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    std::cout << "OpType: EltwiseOp" << std::endl;
    const ::tt::target::ttnn::EltwiseOp *eltwiseOp = op->type_as_EltwiseOp();
    if (operations::unary::isUnaryOp(eltwiseOp)) {
      operations::unary::run(eltwiseOp, context);
    } else {
      assert(operations::binary::isBinaryOp(eltwiseOp) &&
             "Eltwise op should be either unary or binary");
      operations::binary::run(eltwiseOp, context);
    }
    std::cout << "Op finished: EltwiseOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    std::cout << "OpType: MatmulOp" << std::endl;
    operations::matmul::run(op->type_as_MatmulOp(), context);
    std::cout << "Op finished: MatmulOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    std::cout << "OpType: ReductionOp" << std::endl;
    operations::reduction::run(op->type_as_ReductionOp(), context);
    std::cout << "Op finished: ReductionOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    std::cout << "OpType: EmbeddingOp" << std::endl;
    operations::embedding::run(op->type_as_EmbeddingOp(), context);
    std::cout << "Op finished: EmbeddingOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    std::cout << "OpType: SoftmaxOp" << std::endl;
    operations::normalization::run(op->type_as_SoftmaxOp(), context);
    std::cout << "Op finished: SoftmaxOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    std::cout << "OpType: TransposeOp" << std::endl;
    operations::data_movement::run(op->type_as_TransposeOp(), context);
    std::cout << "Op finished: TransposeOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    std::cout << "OpType: ConcatOp" << std::endl;
    operations::data_movement::run(op->type_as_ConcatOp(), context);
    std::cout << "Op finished: ConcatOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    std::cout << "OpType: ReshapeOp" << std::endl;
    operations::data_movement::run(op->type_as_ReshapeOp(), context);
    std::cout << "Op finished: ReshapeOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    std::cout << "OpType: Conv2dOp" << std::endl;
    operations::conv::run(op->type_as_Conv2dOp(), context);
    std::cout << "Op finished: Conv2dOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocOp: {
    std::cout << "OpType: DeallocOp" << std::endl;
    operations::deletion::run(op->type_as_DeallocOp(), context);
    std::cout << "Op finished: DeallocOp" << std::endl;
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    std::cout << "OpType: MaxPool2dOp" << std::endl;
    operations::pool::run(op->type_as_MaxPool2dOp(), context);
    std::cout << "Op finished: MaxPool2dOp" << std::endl;
    break;
  }
  default: {
    throw std::runtime_error("Unsupported operation type");
  }
  }
}

// Nop is single input, output tensor where input is returned as output.
static bool handleNopProgram(::tt::target::ttnn::Program const *program,
                             std::vector<::ttnn::Tensor *> const &inputs,
                             std::vector<::ttnn::Tensor *> const &outputs) {

  bool isNop = program->inputs()->size() == 1 &&
               program->outputs()->size() == 1 &&
               program->inputs()->Get(0)->global_id() ==
                   program->outputs()->Get(0)->global_id();

  if (isNop) {
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(*inputs.at(0));
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(*outputs.at(0));
    std::uint32_t size = outputs[0]->volume() * outputs[0]->element_size();
    std::memcpy(dst, src, size);
  }
  return isNop;
}

void runProgram(::ttnn::Device &device,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  if (handleNopProgram(program, inputs, outputs)) {
    return;
  }
  TensorMap liveTensors;
  DeviceMap allDevices;
  int inputIndex = 0;
  assert(program->inputs()->size() == inputs.size());
  // Assuming single device for now until we support multichip
  allDevices.try_emplace(device.id(), &device);
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    assert(inserted && "Duplicate input tensor");
  }

  int outputIndex = 0;
  assert(program->outputs()->size() == outputs.size());
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(output->global_id(), outputs[outputIndex++]);
    assert(inserted && "Duplicate output tensor");
  }
  ProgramExecutor executor(liveTensors, allDevices);
  executor.execute(program);
}

} // namespace tt::runtime::ttnn
