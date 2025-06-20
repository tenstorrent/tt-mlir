// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-alchemist/tt_alchemist.h"
#include "include/compiler.h"
#include "include/runtime.h"
#include "include/solution_generator.h"

namespace tt_alchemist {

// Implementation class
class TTAlchemistImpl {
public:
  TTAlchemistImpl() : compiler_(), solution_generator_(), runtime_() {}

  Compiler compiler_;
  SolutionGenerator solution_generator_;
  Runtime runtime_;
  std::string last_error_;
};

TTAlchemist::TTAlchemist() : impl_(std::make_unique<TTAlchemistImpl>()) {}

TTAlchemist::~TTAlchemist() = default;

bool TTAlchemist::modelToCpp(const std::string &input_file,
                             const ConversionConfig &config) {
  // Create a temporary file for the C++ output
  std::string temp_cpp_file = config.output_dir + "/temp_model.cpp";

  // Compile the model to C++
  if (!impl_->compiler_.compileToEmitC(input_file, temp_cpp_file,
                                       config.opt_level)) {
    impl_->last_error_ = impl_->compiler_.getLastError();
    return false;
  }

  // Generate the solution
  if (!impl_->solution_generator_.generateSolution(temp_cpp_file,
                                                   config.output_dir)) {
    impl_->last_error_ = impl_->solution_generator_.getLastError();
    return false;
  }

  return true;
}

bool TTAlchemist::buildSolution(const std::string &model_dir,
                                const BuildConfig &config) {
  if (!impl_->runtime_.buildSolution(model_dir, config.flavor, config.target)) {
    impl_->last_error_ = impl_->runtime_.getLastError();
    return false;
  }

  return true;
}

bool TTAlchemist::runSolution(const std::string &model_dir,
                              const RunConfig &config) {
  if (!impl_->runtime_.runSolution(model_dir, config.input_file,
                                   config.output_file)) {
    impl_->last_error_ = impl_->runtime_.getLastError();
    return false;
  }

  return true;
}

bool TTAlchemist::profileSolution(const std::string &model_dir,
                                  const RunConfig &config,
                                  const std::string &report_file) {
  if (!impl_->runtime_.profileSolution(model_dir, config.input_file,
                                       report_file)) {
    impl_->last_error_ = impl_->runtime_.getLastError();
    return false;
  }

  return true;
}

std::string TTAlchemist::getLastError() const { return impl_->last_error_; }

} // namespace tt_alchemist
