// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_FLATBUFFEROBJECTCACHE_H
#define TTMLIR_TARGET_UTILS_FLATBUFFEROBJECTCACHE_H

#include "flatbuffers/flatbuffers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

#include <type_traits>

namespace mlir::tt {

struct FlatbufferObjectCache {
  ::flatbuffers::FlatBufferBuilder *fbb;
  DenseMap<const void *, ::flatbuffers::uoffset_t> objectMap;
  /// BufferDesc entries keyed by buffer SSA Value. Kept separate from
  /// objectMap because the same Value is also used to cache BufferRef; a
  /// single map would collide on insert.
  DenseMap<const void *, ::flatbuffers::uoffset_t> bufferDescByValue;
  uint32_t global_id = 1; // 0 is reserved for null

  FlatbufferObjectCache(::flatbuffers::FlatBufferBuilder *fbb) : fbb(fbb) {}

  template <typename T>
  struct offset_extract_t;
  template <template <typename> typename OffsetT, typename T>
  struct offset_extract_t<OffsetT<T>> {
    using type = T;
  };

  uint32_t nextGlobalId() { return global_id++; }

  template <typename MLIRTypeOrAttr>
  bool exists(MLIRTypeOrAttr obj) const {
    return objectMap.contains(obj.getAsOpaquePointer());
  }

  template <typename MLIRTypeOrAttr, typename SchemaType>
  flatbuffers::Offset<SchemaType>
  insert(MLIRTypeOrAttr obj, flatbuffers::Offset<SchemaType> offset) {
    assert(!exists(obj) && "object already exists");
    objectMap.insert(std::make_pair(obj.getAsOpaquePointer(), offset.o));
    return offset;
  }

  template <typename SchemaType, typename MLIRTypeOrAttr>
  flatbuffers::Offset<SchemaType> at(MLIRTypeOrAttr obj) const {
    assert(exists(obj) && "object does not exist");
    return flatbuffers::Offset<SchemaType>(
        objectMap.at(obj.getAsOpaquePointer()));
  }

  template <typename MLIRTypeOrAttr, typename CreateFn, typename... Args>
  std::invoke_result_t<CreateFn, FlatbufferObjectCache &, MLIRTypeOrAttr,
                       Args...>
  getOrCreate(MLIRTypeOrAttr obj, CreateFn createFn, Args... args) {
    using SchemaType = typename offset_extract_t<std::invoke_result_t<
        CreateFn, FlatbufferObjectCache &, MLIRTypeOrAttr, Args...>>::type;

    if (exists(obj)) {
      return at<SchemaType, MLIRTypeOrAttr>(obj);
    }
    return insert(obj, createFn(*this, obj, args...));
  }

  /// Like getOrCreate but uses bufferDescByValue so Value keys do not collide
  /// with BufferRef (or other) entries in objectMap.
  template <typename CreateFn, typename... Args>
  std::invoke_result_t<CreateFn, FlatbufferObjectCache &, mlir::Value, Args...>
  getOrCreateBufferDescForValue(mlir::Value value, CreateFn createFn,
                                Args... args) {
    using ReturnType = std::invoke_result_t<CreateFn, FlatbufferObjectCache &,
                                            mlir::Value, Args...>;
    using SchemaType = typename offset_extract_t<ReturnType>::type;

    const void *key = value.getAsOpaquePointer();
    if (auto it = bufferDescByValue.find(key); it != bufferDescByValue.end()) {
      return flatbuffers::Offset<SchemaType>(it->second);
    }
    ReturnType offset = createFn(*this, value, args...);
    bufferDescByValue.insert(std::make_pair(key, offset.o));
    return offset;
  }

  template <typename MLIRTypeOrAttr, typename CreateFn, typename... Args>
  std::invoke_result_t<CreateFn, FlatbufferObjectCache &, MLIRTypeOrAttr,
                       mlir::tt::ttcore::ShardStatus, Args...>
  getOrCreateNoSharding(MLIRTypeOrAttr obj, CreateFn createFn, Args &&...args) {
    return getOrCreate(obj, createFn, mlir::tt::ttcore::ShardStatus::Unsharded,
                       std::forward<Args>(args)...);
  }
};

} // namespace mlir::tt

#endif
