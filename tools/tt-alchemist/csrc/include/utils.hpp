// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_UTILS_HPP
#define TT_ALCHEMIST_UTILS_HPP

#include <dlfcn.h>
#include <filesystem>

namespace fs = std::filesystem;

inline std::filesystem::path get_templates_dir() {
  // Templates dir location is relative to the shared library
  //
  Dl_info info;
  dladdr(reinterpret_cast<void *>(&get_templates_dir), &info);
  fs::path so_path = fs::canonical(info.dli_fname);
  return so_path.parent_path().parent_path() / "templates";
}

#endif // TT_ALCHEMIST_UTILS_HPP
