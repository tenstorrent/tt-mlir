// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_PYTHON_UTILS_H
#define TTMLIR_TARGET_PYTHON_UTILS_H

#pragma clang diagnostic push

#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wzero-length-array"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#pragma clang diagnostic pop

namespace nb = nanobind;

namespace mlir::ttmlir::python {

inline nb::capsule wrapInCapsule(std::shared_ptr<void> underlying) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  std::shared_ptr<void> *binary = static_cast<std::shared_ptr<void> *>(
      std::malloc(sizeof(std::shared_ptr<void>)));
  assert(binary);
  *binary = underlying;
  return nb::capsule(
      static_cast<void *>(
          binary), // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      +[](void *data) noexcept { std::free(data); });
}

} // namespace mlir::ttmlir::python

#endif // TTMLIR_TARGET_PYTHON_UTILS_H
