// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_UTILS_HPP
#define TT_ALCHEMIST_UTILS_HPP

#include <dlfcn.h>
#include <filesystem>

namespace mlir {
class ModuleOp;
class PassManager;
} // namespace mlir

namespace tt::alchemist::utils {
namespace fs = std::filesystem;

inline std::filesystem::path get_templates_dir() {
  // Find templates directory in multiple scenarios:
  // 1. Wheel install: site-packages/tt_alchemist/lib/libtt-alchemist-lib.so
  //    → templates at site-packages/tt_alchemist/templates/
  // 2. Editable install: build/lib/libtt-alchemist-lib.so
  //    → templates symlinked at
  //    build/tools/tt-alchemist/wheel/tt_alchemist/templates/
  // 3. Direct build: build/lib/libtt-alchemist-lib.so
  //    → templates at build/tools/tt-alchemist/templates/

  Dl_info info;
  dladdr(reinterpret_cast<void *>(&get_templates_dir), &info);
  fs::path so_path = fs::canonical(info.dli_fname);
  fs::path so_dir = so_path.parent_path();

  // Scenario 1: Wheel install (installed package)
  // .so is at: tt_alchemist/lib/libtt-alchemist-lib.so
  // templates at: tt_alchemist/templates/
  fs::path candidate1 = so_dir.parent_path() / "templates";
  if (fs::exists(candidate1)) {
    return fs::canonical(candidate1);
  }

  // Scenario 2 & 3: Build directory
  // .so is at: build/lib/libtt-alchemist-lib.so
  // Need to find build/tools/tt-alchemist/templates/ or
  //            build/tools/tt-alchemist/wheel/tt_alchemist/templates/

  // Try editable install location first
  fs::path candidate2 = so_dir.parent_path() / "tools" / "tt-alchemist" /
                        "wheel" / "tt_alchemist" / "templates";
  if (fs::exists(candidate2)) {
    return fs::canonical(candidate2);
  }

  // Try direct build location
  fs::path candidate3 =
      so_dir.parent_path() / "tools" / "tt-alchemist" / "templates";
  if (fs::exists(candidate3)) {
    return fs::canonical(candidate3);
  }

  // If nothing found, throw an error with helpful message
  throw std::runtime_error("Could not find templates directory. Searched:\n"
                           "  1. " +
                           candidate1.string() +
                           " (wheel install)\n"
                           "  2. " +
                           candidate2.string() +
                           " (editable install)\n"
                           "  3. " +
                           candidate3.string() +
                           " (direct build)\n"
                           "Library location: " +
                           so_path.string());
}

enum class CodeGenerationTarget { Cpp, Python };

std::string getPipelineName(mlir::ModuleOp module, CodeGenerationTarget target);

bool runPipeline(mlir::PassManager &pm, mlir::ModuleOp module,
                 const std::string &pipelineName,
                 const std::string &pipelineOptions = "");
bool runPipeline(mlir::PassManager &pm, mlir::ModuleOp module,
                 CodeGenerationTarget target,
                 const std::string &pipelineOptions = "");

void formatCode(const fs::path &dirOrFilePath, CodeGenerationTarget target);
} // namespace tt::alchemist::utils

#endif // TT_ALCHEMIST_UTILS_HPP
