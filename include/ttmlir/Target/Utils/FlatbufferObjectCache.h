#ifndef TTMLIR_TARGET_UTILS_FLATBUFFEROBJECTCACHE_H
#define TTMLIR_TARGET_UTILS_FLATBUFFEROBJECTCACHE_H

#include "flatbuffers/flatbuffers.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt {

struct FlatbufferObjectCache {
  ::flatbuffers::FlatBufferBuilder *fbb;
  DenseMap<void const *, ::flatbuffers::uoffset_t> objectMap;
  uint32_t global_id = 1; // 0 is reserved for null

  FlatbufferObjectCache(::flatbuffers::FlatBufferBuilder *fbb) : fbb(fbb) {}

  template <typename T> struct offset_extract_t;
  template <template <typename> typename OffsetT, typename T>
  struct offset_extract_t<OffsetT<T>> {
    using type = T;
  };

  uint32_t nextGlobalId() { return global_id++; }

  template <typename MLIRTypeOrAttr> bool exists(MLIRTypeOrAttr obj) const {
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
};

} // namespace mlir::tt

#endif
