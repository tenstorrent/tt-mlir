// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"

// Get the return type of the toFlatbuffer function.
template <typename T>
using ToFlatbufferReturnType = decltype(toFlatbuffer(
    std::declval<mlir::tt::FlatbufferObjectCache &>(), std::declval<T>()));

// Check if the type T has a toFlatbuffer function.
template <typename T>
concept HasToFlatbuffer =
    requires(mlir::tt::FlatbufferObjectCache &cache, T value) {
      { toFlatbuffer(cache, value) };
    };

// Check if the type T is arithmetic.
template <typename T>
concept IsArithmetic = std::is_arithmetic_v<T>;

// Check if the type T has a Traits::type member.
// Types which are defined in the flatbuffer have this member.
template <typename T>
concept HasTraitsType = requires { typename T::Traits::type; };

// Constrant the template only to types that are native flatbuffer types.
template <typename T>
concept NativeFlatbufferTypeC =
    requires(mlir::tt::FlatbufferObjectCache &cache, T value) {
      { toFlatbuffer(cache, value) } -> HasTraitsType;
    };

// Constraint the template only to types that are not native flatbuffer types.
template <typename T>
concept NonNativeFlatbufferTypeC =
    !NativeFlatbufferTypeC<T> &&
    requires(mlir::tt::FlatbufferObjectCache &cache, T value) {
      { toFlatbuffer(cache, value) };
    };
