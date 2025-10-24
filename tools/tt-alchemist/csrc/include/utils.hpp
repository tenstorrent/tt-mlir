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
  // Templates dir location is relative to the shared library
  //
  Dl_info info;
  dladdr(reinterpret_cast<void *>(&get_templates_dir), &info);
  fs::path so_path = fs::canonical(info.dli_fname);
  return so_path.parent_path().parent_path() / "templates";
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
