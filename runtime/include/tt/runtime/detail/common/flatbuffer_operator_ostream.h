// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_FLATBUFFER_OPERATOR_OSTREAM_H
#define TT_RUNTIME_DETAIL_COMMON_FLATBUFFER_OPERATOR_OSTREAM_H

// This file defines an ostream operator that is generic over our flatbuffer
// types.  Usage:
//   std::cout << *my_flatbuffer->desc()->foo();
//   LOG_INFO("foo: ", *my_flatbuffer->desc()->foo());

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"

#include <ostream>
#include <type_traits>

#include "flatbuffers/idl.h"

#include "tt/runtime/detail/common/logger.h"

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
#include "ttmlir/Target/TTMetal/binary_bfbs_generated.h"
#include "ttmlir/Target/TTMetal/binary_generated.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "ttmlir/Target/TTNN/binary_bfbs_generated.h"
#include "ttmlir/Target/TTNN/binary_generated.h"
#endif

#pragma clang diagnostic pop

namespace tt::runtime::flatbuffer_operator_ostream::detail {
template <typename Schema>
const ::flatbuffers::Parser &getParser() {
  ::flatbuffers::IDLOptions opts;
  opts.size_prefixed = true;
  opts.strict_json = true;
  opts.output_default_scalars_in_json = true;
  static ::flatbuffers::Parser parser(opts);
  static bool initialized = false;
  if (initialized) {
    return parser;
  }
  if (!parser.Deserialize(Schema::data(), Schema::size())) {
    LOG_FATAL("Failed to deserialize schema");
  }
  initialized = true;
  return parser;
}

template <typename Schema, typename NamespacePrefix, typename FlatbufferT>
std::ostream &print(std::ostream &os, const FlatbufferT &f) {
  constexpr bool isQualified =
      std::string_view(FlatbufferT::GetFullyQualifiedName())
          .compare(0, NamespacePrefix::ns.size(), NamespacePrefix::ns) == 0;
  static_assert(isQualified, "flatbuffer must be qualified with namespace "
                             "'tt.target.*.' to use this printer");

  const ::flatbuffers::Parser &parser = getParser<Schema>();
  std::string json;
  const char *err = ::flatbuffers::GenTextFromTable(
      parser, &f, FlatbufferT::GetFullyQualifiedName(), &json);
  LOG_ASSERT(!err, "Failed to generate JSON: ", err);
  os << json;
  return os;
}
} // namespace tt::runtime::flatbuffer_operator_ostream::detail

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
namespace tt::target::metal {
struct NamespacePrefix {
  static constexpr std::string_view ns = "tt.target.metal.";
};

template <typename FlatbufferT,
          typename = std::enable_if_t<std::is_same_v<
              decltype(FlatbufferT::GetFullyQualifiedName()), const char *>>>
std::ostream &operator<<(std::ostream &os, const FlatbufferT &f) {
  return tt::runtime::flatbuffer_operator_ostream::detail::print<
      TTMetalBinaryBinarySchema, NamespacePrefix>(os, f);
}
} // namespace tt::target::metal
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN)
namespace tt::target::ttnn {
struct NamespacePrefix {
  static constexpr std::string_view ns = "tt.target.ttnn.";
};

template <
    typename FlatbufferT,
    typename = std::enable_if_t<std::is_same_v<
        decltype(&FlatbufferT::GetFullyQualifiedName), const char *(*)(void)>>>
std::ostream &operator<<(std::ostream &os, const FlatbufferT &f) {
  return tt::runtime::flatbuffer_operator_ostream::detail::print<
      TTNNBinaryBinarySchema, NamespacePrefix>(os, f);
}
} // namespace tt::target::ttnn
#endif

#endif // TT_RUNTIME_DETAIL_COMMON_FLATBUFFER_OPERATOR_OSTREAM_H
