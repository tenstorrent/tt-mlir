// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DEBUG_H
#define TT_RUNTIME_DETAIL_DEBUG_H

#include <functional>
#include <optional>
#include <ostream>

#include "tt/runtime/types.h"

namespace tt::runtime::debug {

struct Env {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Env const &
#else
  constexpr static Env
#endif
  get(bool loadKernelsFromDisk = false, bool enableAsyncTTNN = false)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false, false);
  }
#endif

  bool loadKernelsFromDisk;
  bool enableAsyncTTNN;

private:
  constexpr Env(bool loadKernelsFromDisk, bool enableAsyncTTNN)
      : loadKernelsFromDisk(loadKernelsFromDisk),
        enableAsyncTTNN(enableAsyncTTNN) {}
};

inline std::ostream &operator<<(std::ostream &os, Env const &env) {
  os << "debug::Env{\n"
     << "\t" << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << ",\n"
     << "\t" << "enableAsyncTTNN: " << env.enableAsyncTTNN << "\n"
     << "}";
  return os;
}

struct RuntimeConfig {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static RuntimeConfig const &
#else
  constexpr static RuntimeConfig
#endif
  get(double atol = 1e-08, double rtol = 1e-05, double pcc = 0.99,
      std::string artifact_dir = "")
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return RuntimeConfig(false, false);
  }
#endif

  double atol;
  double rtol;
  double pcc;
  std::string artifact_dir;

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  RuntimeConfig(double atol, double rtol, double pcc, std::string artifact_dir)
      : atol(atol), rtol(rtol), pcc(pcc), artifact_dir(artifact_dir) {}
#else
  constexpr RuntimeConfig() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os,
                                RuntimeConfig const &runtimeConfig) {
  os << "debug::RuntimeConfig{\n"
     << "\t" << "atol: " << runtimeConfig.atol << ",\n"
     << "\t" << "rtol: " << runtimeConfig.rtol << ",\n"
     << "\t" << "pcc: " << runtimeConfig.pcc << ",\n"
     << "\t" << "artifact_dir: " << runtimeConfig.artifact_dir << "\n"
     << "}";
  return os;
}

struct Hooks {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Hooks const &
  get(std::optional<std::function<void(RuntimeConfig, Binary, CallbackContext,
                                       OpContext)>>
          operatorCallback = std::nullopt);
#else
  constexpr static Hooks get() { return Hooks(); }
#endif

  std::optional<
      std::function<void(RuntimeConfig, Binary, CallbackContext, OpContext)>>
  getOperatorCallback() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return operatorCallback;
#else
    return std::nullopt;
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Hooks(std::optional<
        std::function<void(RuntimeConfig, Binary, CallbackContext, OpContext)>>
            operatorCallback)
      : operatorCallback(operatorCallback) {}

  std::optional<
      std::function<void(RuntimeConfig, Binary, CallbackContext, OpContext)>>
      operatorCallback;
#else
  constexpr Hooks() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, Hooks const &hooks) {
  os << "debug::Hooks{\n"
     << "\t"
     << "operatorCallback: " << static_cast<bool>(hooks.getOperatorCallback())
     << ",\n"
     << "}";
  return os;
}

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
