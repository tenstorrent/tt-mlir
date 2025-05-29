// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_PYTHON_BINARY_BINARY_H
#define TT_RUNTIME_PYTHON_BINARY_BINARY_H

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace tt::runtime::python {
namespace nb = nanobind;
void registerBinaryBindings(nb::module_ &m);
} // namespace tt::runtime::python

#endif // TT_RUNTIME_PYTHON_BINARY_BINARY_H
