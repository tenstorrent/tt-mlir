// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_HPP
#define TT_ALCHEMIST_HPP

#include <mlir/IR/MLIRContext.h>

#include <string>
#include <vector>

namespace tt::alchemist {

// Test generation options for unit test generation
struct TestGenerationOptions {
  std::vector<std::string> opFilter;     // Filter for specific operations (empty = all)
  bool generateParametrized = true;      // Generate parametrized tests
  std::string testFramework = "pytest";  // Test framework to use
  bool useEmitPy = true;                 // Use EmitPy for Python code generation
  std::string pipelineOptions = "";      // Additional MLIR pipeline options
  bool generateConftest = true;          // Generate conftest.py with fixtures
  bool verbose = false;                  // Verbose output during generation
};

// Main interface for the tt-alchemist tool.
//
// This header defines the TTAlchemist singleton class, which provides methods
// to convert MLIR models to C++ or Python code, and to generate standalone
// solutions from these models. The class manages an MLIRContext internally.
//
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
                   bool is_local = true,
                   const std::string &pipeline_options = "");

  // Generate a standalone solution with the generated Python code
  bool generatePython(const std::string &input_file,
                      const std::string &output_dir, bool is_local = true,
                      const std::string &pipeline_options = "");

  // Generate unit tests from MLIR file
  bool generateUnitTests(const std::string &input_file,
                        const std::string &output_dir,
                        const TestGenerationOptions &options = {});

  // Generate unit tests from MLIR module string
  bool generateUnitTestsFromString(const std::string &mlir_string,
                                   const std::string &output_dir,
                                   const TestGenerationOptions &options = {});

private:
  TTAlchemist();

  mlir::MLIRContext context;
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_HPP
