// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <memory>

// Define the expected signature for the add function based on the IR
// The function returns a struct with pointers and metadata.
using AddFn = void *(*)(float *, float *, int64_t, int64_t, int64_t, int64_t,
                        int64_t, float *, float *, int64_t, int64_t, int64_t,
                        int64_t, int64_t);

// Ensure correct memory alignment
constexpr size_t matrix_size = 32 * 32 * sizeof(float);
constexpr int64_t tensor_size = 32;
constexpr int64_t alignment =
    64; // Adjust if needed, often 64 is good for vectorized operations

int main() {
  // Load the shared library (adjust path to generated.so)
  void *handle = dlopen("./generated.so", RTLD_NOW);
  if (!handle) {
    std::cerr << "Failed to load library: " << dlerror() << std::endl;
    return 1;
  }

  // Load the add function from the shared library
  AddFn add = (AddFn)dlsym(handle, "add");
  if (!add) {
    std::cerr << "Failed to load function 'add': " << dlerror() << std::endl;
    dlclose(handle);
    return 1;
  }

  // Allocate and initialize the input tensors (32x32 floats)
  float *tensor1 = (float *)std::aligned_alloc(alignment, matrix_size);
  float *tensor2 = (float *)std::aligned_alloc(alignment, matrix_size);
  float *output = (float *)std::aligned_alloc(alignment, matrix_size);

  if (!tensor1 || !tensor2 || !output) {
    std::cerr << "Memory allocation failed" << std::endl;
    dlclose(handle);
    return 1;
  }

  // Initialize tensor1 and tensor2 with some values
  for (int i = 0; i < 32 * 32; i++) {
    tensor1[i] = static_cast<float>(i);     // Initialize with some test data
    tensor2[i] = static_cast<float>(2 * i); // Initialize with some test data
  }

  // Call the add function
  auto result =
      add(tensor1, tensor2, 1, 1, 1, 1, 1, output, output, 1, 1, 1, 1, 1);

  if (!result) {
    std::cerr << "Function 'add' returned a null pointer." << std::endl;
    std::free(tensor1);
    std::free(tensor2);
    std::free(output);
    dlclose(handle);
    return 1;
  }

  // Print some of the result tensor to verify the addition
  std::cout << "Output tensor (first 10 elements):" << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  // Clean up
  std::free(tensor1);
  std::free(tensor2);
  std::free(output);
  dlclose(handle);

  return 0;
}
