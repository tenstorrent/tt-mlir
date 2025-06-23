// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_H
#define TT_ALCHEMIST_H

// Define export macro for C API
#ifndef TT_ALCHEMIST_EXPORT
#define TT_ALCHEMIST_EXPORT __attribute__((visibility("default")))
#endif

#include "mlir/IR/MLIRContext.h"

#include <string>

namespace tt::alchemist {

class TTAlchemist {
public:
  // Singleton pattern
  static TTAlchemist &getInstance() {
    static TTAlchemist instance;
    return instance;
  }

  // Delete copy and move constructors/assignments
  TTAlchemist(const TTAlchemist &) = delete;
  TTAlchemist &operator=(const TTAlchemist &) = delete;
  TTAlchemist(TTAlchemist &&) = delete;
  TTAlchemist &operator=(TTAlchemist &&) = delete;

  bool modelToCpp(const std::string &input_file);

  // Create a standalone solution with the generated C++ code
  bool createSolution(const std::string &input_file,
                      const std::string &output_dir);

private:
  TTAlchemist();

  mlir::MLIRContext context;
};

} // namespace tt::alchemist

// C-compatible API for external usage
#ifdef __cplusplus
extern "C" {
#endif

// Get the singleton instance
TT_ALCHEMIST_EXPORT void *tt_alchemist_TTAlchemist_getInstance();

// Model to CPP conversion
TT_ALCHEMIST_EXPORT bool
tt_alchemist_TTAlchemist_modelToCpp(void *instance, const char *input_file);

// Create standalone solution
TT_ALCHEMIST_EXPORT bool
tt_alchemist_TTAlchemist_createSolution(void *instance, const char *input_file,
                                        const char *output_dir);

#ifdef __cplusplus
}
#endif

#endif // TT_ALCHEMIST_H
