// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTKernelTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir::tt::ttkernel;

MlirType ttmlirTTKernelCBTypeGet(MlirContext ctx, MlirType memrefType) {
  return wrap(CBType::get(mlir::cast<mlir::MemRefType>(unwrap(memrefType))));
}

MlirType ttmlirTTKernelSemaphoreTypeGet(MlirContext ctx) {
  return wrap(SemaphoreType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelNocAddrTypeGet(MlirContext ctx) {
  return wrap(NocAddrType::get(unwrap(ctx)));
}

MlirAttribute ttmlirTTKernelThreadTypeAttrGet(MlirContext ctx,
                                              uint32_t enumValue) {
  return wrap(
      ThreadTypeAttr::get(unwrap(ctx), static_cast<ThreadType>(enumValue)));
}

MlirAttribute ttmlirTTKernelReduceTypeAttrGet(MlirContext ctx,
                                              uint32_t enumValue) {
  return wrap(
      ReduceTypeAttr::get(unwrap(ctx), static_cast<ReduceType>(enumValue)));
}

MlirAttribute ttmlirTTKernelReduceDimAttrGet(MlirContext ctx,
                                             uint32_t enumValue) {
  return wrap(
      ReduceDimAttr::get(unwrap(ctx), static_cast<ReduceDim>(enumValue)));
}

MlirType ttmlirTTKernelL1AddrTypeGet(MlirContext ctx) {
  return wrap(L1AddrType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelL1AddrPtrTypeGet(MlirContext ctx) {
  return wrap(L1AddrPtrType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelInterleavedAddrGenFastTypeGet(MlirContext ctx) {
  return wrap(InterleavedAddrGenFastType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelDataFormatTypeGet(MlirContext ctx) {
  return wrap(DataFormatType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelTensorAccessorArgsTypeGet(MlirContext ctx) {
  return wrap(TensorAccessorArgsType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelTensorAccessorTypeGet(MlirContext ctx) {
  return wrap(TensorAccessorType::get(unwrap(ctx)));
}

MlirType ttmlirTTKernelTensorAccessorPageMappingTypeGet(MlirContext ctx) {
  return wrap(TensorAccessorPageMappingType::get(unwrap(ctx)));
}

MlirAttribute ttmlirTTKernelArgAttrGet(MlirContext ctx, uint32_t argTypeValue,
                                       size_t operandIndex, bool isUniform) {
  return wrap(ArgAttr::get(unwrap(ctx), static_cast<ArgType>(argTypeValue),
                           operandIndex));
}

MlirAttribute ttmlirTTKernelArgSpecAttrGet(MlirContext ctx,
                                           MlirAttribute *rtArgs,
                                           size_t rtArgsSize,
                                           MlirAttribute *ctArgs,
                                           size_t ctArgsSize) {
  std::vector<ArgAttr> _rt_args, _ct_args;

  for (size_t i = 0; i < rtArgsSize; i++) {
    _rt_args.emplace_back(mlir::cast<ArgAttr>(unwrap(rtArgs[i])));
  }

  for (size_t i = 0; i < ctArgsSize; i++) {
    _ct_args.emplace_back(mlir::cast<ArgAttr>(unwrap(ctArgs[i])));
  }

  return wrap(ArgSpecAttr::get(unwrap(ctx), _rt_args, _ct_args));
}
