// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_HPP
#define TT_ALCHEMIST_HPP

#include <mlir/IR/MLIRContext.h>

#include <string>

namespace tt::alchemist {

class TTAlchemist {
public:
  // Singleton pattern
  static TTAlchemist &getInstance();

  // Delete copy and move constructors/assignments
  TTAlchemist(const TTAlchemist &) = delete;
  TTAlchemist &operator=(const TTAlchemist &) = delete;
  TTAlchemist(TTAlchemist &&) = delete;
  TTAlchemist &operator=(TTAlchemist &&) = delete;

  // Convert MLIR model to C++ code
  bool modelToCpp(const std::string &input_file);

  // Convert MLIR model to Python code
  bool modelToPython(const std::string &input_file);

  // Generate a standalone solution with the generated C++ code
  bool generateCpp(const std::string &input_file, const std::string &output_dir,
                   bool is_local = true);

  // Generate a standalone solution with the generated Python code
  bool generatePython(const std::string &input_file,
                      const std::string &output_dir, bool is_local = true);

private:
  TTAlchemist();

  mlir::MLIRContext context;
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_HPP
