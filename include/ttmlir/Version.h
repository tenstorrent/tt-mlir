// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_VERSION_H
#define TTMLIR_VERSION_H

namespace ttmlir {

#ifndef TTMLIR_GIT_HASH
#define TTMLIR_GIT_HASH "unknown"
// #error "TTMLIR_GIT_HASH must be defined"
#endif
#ifndef TTMLIR_VERSION_MAJOR
#define TTMLIR_VERSION_MAJOR 0
// #error "TTMLIR_VERSION_MAJOR must be defined"
#endif
#ifndef TTMLIR_VERSION_MINOR
#define TTMLIR_VERSION_MINOR 0
// #error "TTMLIR_VERSION_MINOR must be defined"
#endif
#ifndef TTMLIR_VERSION_PATCH
#define TTMLIR_VERSION_PATCH 0
// #error "TTMLIR_VERSION_PATCH must be defined"
#endif

struct Version {
  unsigned major;
  unsigned minor;
  unsigned patch;

  constexpr Version(unsigned major, unsigned minor, unsigned patch)
      : major(major), minor(minor), patch(patch) {}

  constexpr bool operator<=(const Version &other) const {
    return major < other.major ||
           (major == other.major && (minor <= other.minor));
  }

  constexpr bool operator>=(const Version &other) const {
    return major > other.major ||
           (major == other.major && (minor >= other.minor));
  }
};

inline constexpr Version getVersion() {
  return Version(TTMLIR_VERSION_MAJOR, TTMLIR_VERSION_MINOR,
                 TTMLIR_VERSION_PATCH);
}

#define XSTR(s) STR(s)
#define STR(s) #s
inline constexpr const char *getGitHash() { return XSTR(TTMLIR_GIT_HASH); }
#undef STR
#undef XSTR

} // namespace ttmlir
#endif
