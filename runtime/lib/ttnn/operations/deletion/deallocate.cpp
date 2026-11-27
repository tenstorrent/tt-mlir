// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/deletion/deallocate.h"
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocateOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensorPool.getTTNNTensorWrapperAndValidate(op->in());
  ::ttnn::Tensor &ttnnTensor = tensorWrapper.getTensor();

  if (std::getenv("QSR_DUMP_DEALLOC") != nullptr) {
    // [ALIASDUMP] A view output shares the INPUT's device address (tt_metal::view builds a new
    // MeshBuffer at input_buffer.address()). If the input is deallocated while a view of it is
    // still live, the view reads freed memory -- which matches reshape_004's signature exactly
    // (wrong values absent from the input, partial, deterministic).
    uint32_t addr = 0;
    bool on_dev = ttnnTensor.storage_type() == ::ttnn::StorageType::DEVICE;
    if (on_dev) { addr = ttnnTensor.mesh_buffer().address(); }
    std::printf("[ALIASDUMP] dealloc tensor addr=0x%x on_device=%d retain=%d force=%d shape=%s\n",
                addr, (int)on_dev, (int)tensorWrapper.shouldRetain(), (int)op->force(),
                [&]{ std::ostringstream o; o << ttnnTensor.logical_shape(); return o.str(); }().c_str());
    std::fflush(stdout);
  }
  // QSR_NO_DEALLOC (2026-09-24): keep every intermediate alive so no device address is ever
  // recycled. forge's ALLOCLOG shows conv2d_011's input landing on 0x138080 -- the address freed
  // from the graph input two ops earlier -- while the pure-ttnn chain that PASSES (0.99994600)
  // holds a Python reference to every intermediate and therefore never reuses an address. Holding
  // a copy of the Tensor keeps its storage alive (shared_ptr), so erasing from the pool below no
  // longer frees the buffer. Memory cost only; unset = no effect.
  if (std::getenv("QSR_NO_DEALLOC") != nullptr) {
    static std::vector<::ttnn::Tensor> keepAlive;
    keepAlive.push_back(ttnnTensor);
    tensorPool.erase(op->in());
    return;
  }
  if (!tensorWrapper.shouldRetain()) {
    ::ttnn::deallocate(ttnnTensor, op->force());
  } else {
    LOG_DEBUG("Tensor is retained thus not deallocating. To deallocate, set "
              "retain to false first");
  }

  tensorPool.erase(op->in());
}
} // namespace tt::runtime::ttnn::operations::deletion
