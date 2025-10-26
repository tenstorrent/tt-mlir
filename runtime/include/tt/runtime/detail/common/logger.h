// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_LOGGER_H
#define TT_RUNTIME_DETAIL_COMMON_LOGGER_H

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <execinfo.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) &&                           \
    (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 1)
#include <pybind11/iostream.h>
#endif

namespace tt::runtime::logger {

namespace detail {
inline std::string demangle(const char *str) {
  size_t size = 0;
  int status = 0;
  std::string rt(256, '\0');
  if (1 == sscanf(str, "%*[^(]%*[^_]%255[^)+]", &rt[0])) {
    std::unique_ptr<char, decltype(&free)> v(
        abi::__cxa_demangle(&rt[0], nullptr, &size, &status), free);
    if (v) {
      std::string result(v.get());
      return result;
    }
  }
  return str;
}

// https://www.fatalerrors.org/a/backtrace-function-and-assert-assertion-macro-encapsulation.html
/**
 * @brief Get the current call stack
 * @param[out] bt Save Call Stack
 * @param[in] size Maximum number of return layers
 * @param[in] skip Skip the number of layers at the top of the stack
 */
inline std::vector<std::string> backtrace(int size = 64, int skip = 1) {
  std::vector<std::string> bt;
  std::vector<void *> array(size);

  int s = ::backtrace(array.data(), size);

  // https://linux.die.net/man/3/backtrace_symbols
  // The address of the array of string pointers is returned as the function
  // result of backtrace_symbols(). This array is malloc(3)ed by
  // backtrace_symbols(), and must be freed by the caller. (The strings pointed
  // to by the array of pointers need not and should not be freed.)
  std::unique_ptr<char *, decltype(&free)> strings(
      backtrace_symbols(array.data(), s), free);

  if (!strings) {
    std::cout << "backtrace_symbols error." << std::endl;
    return bt;
  }

  for (int i = skip; i < s; ++i) {
    bt.push_back(demangle(strings.get()[i]));
  }

  return bt;
}

/**
 * @brief String to get current stack information
 * @param[in] size Maximum number of stacks
 * @param[in] skip Skip the number of layers at the top of the stack
 * @param[in] prefix Output before stack information
 */
inline std::string backtrace_to_string(int size = 64, int skip = 2,
                                       const std::string &prefix = "") {
  std::vector<std::string> bt = backtrace(size, skip);
  std::stringstream ss;
  for (size_t i = 0; i < bt.size(); ++i) {
    ss << prefix << bt[i] << std::endl;
  }
  return ss.str();
}
} // namespace detail

#define LOGGER_TYPES                                                           \
  X(Always)                                                                    \
  X(RuntimeTTNN)                                                               \
  X(RuntimeTTMetal)                                                            \
  X(RuntimeTTMetalBufferCreation)                                              \
  X(RuntimeTTMetalCircularBufferCreation)                                      \
  X(RuntimeTTMetalKernel)                                                      \
  X(RuntimeTTMetalKernelArg)                                                   \
  X(RuntimeTTMetalKernelSource)                                                \
  X(RuntimeTTMetalCommand)

enum LogType : uint32_t {
#define X(a) Log##a,
  LOGGER_TYPES
#undef X
      LogType_Count,
};

static_assert(LogType_Count < 64, "Exceeded number of log types");
static constexpr const char *reset_text_attrs = "\033[0m";
static constexpr const char *bold = "\033[1m";
static constexpr const char *gray = "\033[38;5;240m";
static constexpr const char *cyan = "\033[36m";
static constexpr const char *cornflower_blue = "\033[38;5;69m";
static constexpr const char *green = "\033[32m";
static constexpr const char *orange = "\033[38;5;208m";
static constexpr const char *red = "\033[31m";
class Logger {
public:
  enum class Level {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Fatal = 5,
    Count,
  };

  static constexpr const char *level_names[] = {"TRACE",   "DEBUG", "INFO",
                                                "WARNING", "ERROR", "FATAL"};

  static constexpr const char *type_names[] = {
#define X(a) #a,
      LOGGER_TYPES
#undef X
  };

#undef LOGGER_TYPES

  // All level colors will be bolded
  static constexpr const char *level_colors[] = {
      gray,            // Trace: Gray
      gray,            // Debug: Gray
      cornflower_blue, // Info: Cornflower Blue
      orange,          // Warning: Yellow
      red,             // Error: Red
      red              // Fatal: Red
  };

  static_assert((sizeof(level_colors) / sizeof(level_colors[0])) ==
                static_cast<std::underlying_type_t<Level>>(Level::Count));

  static inline Logger &get() {
    static Logger logger;
    return logger;
  }

  template <typename... Args>
  void log_level_type(Level level, LogType type, const Args &...args) {
    if (log_level_enabled(level) && log_type_enabled(type)) {
#if defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) &&                           \
    (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 1)
      pybind11::scoped_ostream_redirect stream(*fd);
#endif
      *fd << green << std::setw(23) << type_names[type] << reset_text_attrs
          << " | " << bold << level_colors[static_cast<int>(level)]
          << std::setw(8) << level_names[static_cast<int>(level)]
          << reset_text_attrs << " | ";
      ((*fd << args), ...);
      *fd << std::endl;
    }
  }

  void flush() { fd->flush(); }

  bool log_level_enabled(Level level) const {
    return static_cast<int>(level) >= static_cast<int>(min_level);
  }
  bool log_type_enabled(LogType type) const { return (1ULL << type) & mask; }

private:
  Logger() {
    const char *env = std::getenv("TTMLIR_RUNTIME_LOGGER_TYPES");
    if (env) {
      if (strstr(env, "All")) {
        mask = 0xFFFFFFFFFFFFFFFF;
      } else {
        for (uint32_t i = 0; i < LogType_Count; ++i) {
          mask |= (strstr(env, type_names[i]) != nullptr) << i;
        }
      }
    } else {
      mask = 0xFFFFFFFFFFFFFFFF;
    }

    const char *level_env = std::getenv("TTMLIR_RUNTIME_LOGGER_LEVEL");
    if (level_env) {
      std::string level_str = level_env;
      std::transform(level_str.begin(), level_str.end(), level_str.begin(),
                     [](unsigned char c) { return std::toupper(c); });
      for (int i = 0; i < static_cast<int>(Level::Count); i++) {
        if (level_str == level_names[i]) {
          min_level = static_cast<Level>(i);
          break;
        }
      }
    }
#if !defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) ||                          \
    (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 0)
    const char *file_env = std::getenv("TTMLIR_RUNTIME_LOGGER_FILE");
    if (file_env) {
      log_file.open(file_env);
      if (log_file.is_open()) {
        fd = &log_file;
      }
    }
#endif
  }

  static std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto ms = now_ms.time_since_epoch().count() % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%F %T") << '.'
       << std::setfill('0') << std::setw(3) << ms;
    return ss.str();
  }

  std::ofstream log_file;
  std::ostream *fd = &std::cout;
  uint64_t mask = (1ULL << LogAlways);
  Level min_level = Level::Info;
};

template <typename... Args>
inline void log_debug_(LogType type, const Args &...args) {
  Logger::get().log_level_type(Logger::Level::Debug, type, args...);
}

template <typename... Args>
inline void log_debug_(const Args &...args) {
  log_debug_(LogAlways, args...);
}

template <typename... Args>
inline void log_trace_(LogType type, const std::string &src_info,
                       const Args &...args) {
  Logger::get().log_level_type(Logger::Level::Trace, type, src_info, " - ",
                               args...);
}

template <typename... Args>
inline void log_info_(LogType type, const Args &...args) {
  Logger::get().log_level_type(Logger::Level::Info, type, args...);
}

template <typename... Args>
inline void log_info_(const Args &...args) {
  log_info_(LogAlways, args...);
}

template <typename... Args>
inline void log_warning_(LogType type, const Args &...args) {
  Logger::get().log_level_type(Logger::Level::Warning, type, args...);
}

template <typename... Args>
inline void log_warning_(const Args &...args) {
  log_warning_(LogAlways, args...);
}

template <typename... Args>
inline void log_error_(LogType type, const Args &...args) {
  Logger::get().log_level_type(Logger::Level::Error, type, args...);
}

template <typename... Args>
inline void log_error_(const Args &...args) {
  log_error_(LogAlways, args...);
}

template <typename... Args>
inline void log_fatal_(LogType type, const Args &...args) {
  Logger::get().log_level_type(Logger::Level::Fatal, type, args...);
}

template <typename... Args>
inline void log_fatal_(const Args &...args) {
  log_fatal_(LogAlways, args...);
}

template <typename... Args>
[[noreturn]] inline void
tt_throw_(const char *file, int line, const char *assert_type,
          const char *condition_str, const Args &...args) {
  std::stringstream trace_message_ss = {};
  trace_message_ss << "\n";
  trace_message_ss << bold << assert_type << " @ " << file << ":" << line
                   << ": " << condition_str << std::endl;
  trace_message_ss << "backtrace:\n";
  trace_message_ss << detail::backtrace_to_string(100, 3, " --- ");
  trace_message_ss << reset_text_attrs << std::flush;
  log_fatal_(args..., trace_message_ss.str());
  Logger::get().flush();
#if defined(TT_RUNTIME_DEBUG) && (TT_RUNTIME_DEBUG == 1)
  const char *abort_on_error = std::getenv("TTMLIR_RUNTIME_ABORT_ON_ERROR");
  if (abort_on_error && abort_on_error[0] == '1') {
    abort();
  }
#endif
  throw std::runtime_error("Fatal error");
}

} // namespace tt::runtime::logger

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#if defined(TT_RUNTIME_DEBUG) && (TT_RUNTIME_DEBUG == 1)
#define LOG_TRACE(log_type, ...)                                               \
  do {                                                                         \
    if (::tt::runtime::logger::Logger::get().log_level_enabled(                \
            ::tt::runtime::logger::Logger::Level::Trace) &&                    \
        ::tt::runtime::logger::Logger::get().log_type_enabled(log_type))       \
      ::tt::runtime::logger::log_trace_(                                       \
          log_type,                                                            \
          ::tt::runtime::logger::bold + std::string(__FILE__) + ":" +          \
              std::to_string(__LINE__) +                                       \
              ::tt::runtime::logger::reset_text_attrs,                         \
          ##__VA_ARGS__);                                                      \
  } while (0)

#define LOG_DEBUG(...)                                                         \
  do {                                                                         \
    if (::tt::runtime::logger::Logger::get().log_level_enabled(                \
            ::tt::runtime::logger::Logger::Level::Debug)) {                    \
      ::tt::runtime::logger::log_debug_(__VA_ARGS__);                          \
    }                                                                          \
  } while (0)
#else
#define LOG_TRACE(...) ((void)0)
#define LOG_DEBUG(...) ((void)0)
#endif

#define LOG_INFO(...)                                                          \
  do {                                                                         \
    if (::tt::runtime::logger::Logger::get().log_level_enabled(                \
            ::tt::runtime::logger::Logger::Level::Info)) {                     \
      ::tt::runtime::logger::log_info_(__VA_ARGS__);                           \
    }                                                                          \
  } while (0)

#define LOG_WARNING(...)                                                       \
  do {                                                                         \
    if (::tt::runtime::logger::Logger::get().log_level_enabled(                \
            ::tt::runtime::logger::Logger::Level::Warning)) {                  \
      ::tt::runtime::logger::log_warning_(__VA_ARGS__);                        \
    }                                                                          \
  } while (0)

#define LOG_WARNING_ONCE(...)                                                  \
  do {                                                                         \
    if (::tt::runtime::logger::Logger::get().log_level_enabled(                \
            ::tt::runtime::logger::Logger::Level::Warning)) {                  \
      static bool once = false;                                                \
      if (!once) {                                                             \
        once = true;                                                           \
        ::tt::runtime::logger::log_warning_(__VA_ARGS__);                      \
      }                                                                        \
    }                                                                          \
  } while (0)

#define LOG_ERROR(...)                                                         \
  do {                                                                         \
    if (::tt::runtime::logger::Logger::get().log_level_enabled(                \
            ::tt::runtime::logger::Logger::Level::Error)) {                    \
      ::tt::runtime::logger::log_error_(__VA_ARGS__);                          \
    }                                                                          \
  } while (0)

#define LOG_FATAL(...)                                                         \
  do {                                                                         \
    ::tt::runtime::logger::tt_throw_(__FILE__, __LINE__, "LOG_FATAL",          \
                                     "tt::exception", ##__VA_ARGS__);          \
    __builtin_unreachable();                                                   \
  } while (0)

#define LOG_ASSERT(condition, ...)                                             \
  do {                                                                         \
    if (!(condition)) [[unlikely]] {                                           \
      ::tt::runtime::logger::tt_throw_(__FILE__, __LINE__, "LOG_ASSERT",       \
                                       #condition, ##__VA_ARGS__);             \
      __builtin_unreachable();                                                 \
    }                                                                          \
  } while (0)

#if defined(TT_RUNTIME_DEBUG) && (TT_RUNTIME_DEBUG == 1)
#define DEBUG_ASSERT(condition, ...)                                           \
  do {                                                                         \
    if (!(condition)) [[unlikely]] {                                           \
      ::tt::runtime::logger::tt_throw_(__FILE__, __LINE__, "DEBUG_ASSERT",     \
                                       #condition, ##__VA_ARGS__);             \
      __builtin_unreachable();                                                 \
    }                                                                          \
  } while (0)
#else
#define DEBUG_ASSERT(...) ((void)0)
#endif

#pragma clang diagnostic pop

// Helper pretty printers.
namespace tt::runtime::logger {

template <typename T, typename Data>
struct Tag {
  static constexpr std::string_view tag() { return T::tag(); }
  static constexpr std::string_view open() { return T::open(); }
  static constexpr std::string_view close() { return T::close(); }
  static constexpr std::ios_base::fmtflags fmtflags() { return T::fmtflags(); }

  Tag(Data d) : data(d) {}

  Data data;
};

template <typename T, typename Data>
std::ostream &operator<<(std::ostream &os, const Tag<T, Data> &t) {
  os << t.tag() << t.open();
  auto flags = os.setf(t.fmtflags());
  os << t.data;
  os.flags(flags);
  os << t.close();
  return os;
}

template <typename T>
struct HexTag {
  static constexpr std::string_view tag() { return T::tag(); }
  static constexpr std::string_view open() { return "[0x"; }
  static constexpr std::string_view close() { return "]"; }
  static constexpr std::ios_base::fmtflags fmtflags() {
    return std::ios_base::hex;
  }
};

template <typename T>
struct IntegerTag {
  static constexpr std::string_view tag() { return T::tag(); }
  static constexpr std::string_view open() { return "["; }
  static constexpr std::string_view close() { return "]"; }
  static constexpr std::ios_base::fmtflags fmtflags() {
    return std::ios_base::dec;
  }
};

struct AddressTag : HexTag<AddressTag> {
  static constexpr std::string_view tag() { return "address"; }
};

struct AlignTag : HexTag<AlignTag> {
  static constexpr std::string_view tag() { return "align"; }
};

struct BufferTag : IntegerTag<BufferTag> {
  static constexpr std::string_view tag() { return "buffer"; }
};

struct PortTag : IntegerTag<PortTag> {
  static constexpr std::string_view tag() { return "port"; }
};

struct TensorTag : IntegerTag<TensorTag> {
  static constexpr std::string_view tag() { return "tensor"; }
};

struct SizeTag : IntegerTag<SizeTag> {
  static constexpr std::string_view tag() { return "size"; }
};

template <typename IntType>
auto Address(IntType address) {
  return Tag<AddressTag, IntType>(address);
}

template <typename IntType>
auto Size(IntType size) {
  return Tag<SizeTag, IntType>(size);
}

template <typename IntType>
auto Align(IntType align) {
  return Tag<AlignTag, IntType>(align);
}

template <typename IntType>
auto Buffer(IntType buffer) {
  return Tag<BufferTag, IntType>(buffer);
}

template <typename IntType>
auto Port(IntType buffer) {
  return Tag<PortTag, IntType>(buffer);
}

template <typename IntType>
auto Tensor(IntType tensor) {
  return Tag<TensorTag, IntType>(tensor);
}

} // namespace tt::runtime::logger

#endif // TT_RUNTIME_DETAIL_COMMON_LOGGER_H
