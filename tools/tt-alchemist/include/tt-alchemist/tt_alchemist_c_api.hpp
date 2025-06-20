// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_C_API_HPP
#define TT_ALCHEMIST_C_API_HPP

// Define export macro for C API
#ifndef TT_ALCHEMIST_EXPORT
#define TT_ALCHEMIST_EXPORT __attribute__((visibility("default")))
#endif

// C-compatible API for external usage
#ifdef __cplusplus
extern "C" {
#endif

// Get the singleton instance of TTAlchemist
TT_ALCHEMIST_EXPORT void *tt_alchemist_TTAlchemist_getInstance();

// Model to CPP conversion
TT_ALCHEMIST_EXPORT bool
tt_alchemist_TTAlchemist_modelToCpp(void *instance, const char *input_file);

// Generate a standalone solution
TT_ALCHEMIST_EXPORT bool
tt_alchemist_TTAlchemist_generate(void *instance, const char *input_file,
                                  const char *output_dir);

#ifdef __cplusplus
}
#endif

#endif // TT_ALCHEMIST_C_API_HPP
