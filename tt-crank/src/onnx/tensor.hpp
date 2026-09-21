// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt/runtime/types.h>

#include "onnxruntime_c_api.h"

namespace tt::crank::onnx {

// Tensor box.
// It is an optional tensor stored withing OrtValue tensor handle.
struct TensorBox {
    std::optional<::tt::runtime::Tensor> tensor;
};

TensorBox *box_of(const OrtValue *value);

::tt::runtime::Tensor &tensor_of(const OrtValue *value);

bool is_tt(const OrtMemoryDevice *device);
bool is_tt(const OrtValue *val);

// Copies src tensor to dst.
// This function should be called only if any of src and dst is tt tensor.
void copy_tensor(const OrtValue *src, OrtValue *dst);

} // namespace tt::crank::onnx
