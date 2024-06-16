// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_VERSION_H
#define TTMLIR_VERSION_H

namespace ttmlir {

#ifndef TTMLIR_GIT_HASH
#error "TTMLIR_GIT_HASH must be defined"
#endif
#ifndef TTMLIR_VERSION_MAJOR
#error "TTMLIR_VERSION_MAJOR must be defined"
#endif
#ifndef TTMLIR_VERSION_MINOR
#error "TTMLIR_VERSION_MINOR must be defined"
#endif
#ifndef TTMLIR_VERSION_RELEASE
#error "TTMLIR_VERSION_RELEASE must be defined"
#endif

struct Version {
  unsigned major;
  unsigned minor;
  unsigned release;

  constexpr Version(unsigned major, unsigned minor, unsigned release)
      : major(major), minor(minor), release(release) {}

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
                 TTMLIR_VERSION_RELEASE);
}

#define XSTR(s) STR(s)
#define STR(s) #s
inline constexpr const char *getGitHash() { return XSTR(TTMLIR_GIT_HASH); }
#undef STR
#undef XSTR

} // namespace ttmlir
#endif
