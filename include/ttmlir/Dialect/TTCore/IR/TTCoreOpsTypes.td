// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCore_IR_TTCOREOPSTYPES_TD
#define TTMLIR_DIALECT_TTCore_IR_TTCOREOPSTYPES_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreBase.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsEnums.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreAttrInterfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// TT attr definitions
//===----------------------------------------------------------------------===//
// Should Attr be a separate file?

class TTCore_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTCore_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttcore." # attrMnemonic;
}

def TTCore_GridAttr : TTCore_Attr<"Grid", "grid"> {
  let summary = "TT grid attribute";
  let description = [{
    TT grid attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
                        DefaultValuedParameter<
                          "AffineMap",
                          "$_builder.getEmptyAffineMap()">:$mapping);
  let assemblyFormat = "`<` custom<DimensionList>($shape) (`,` $mapping^)? `>`";

  let extraClassDeclaration = [{
      static GridAttr get(::mlir::MLIRContext *context) {
        return GridAttr::get(context, {1, 1});
      }

      static GridAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape) {
        return GridAttr::get(context, shape, AffineMap::get(context));
      }

      static GridAttr get(::mlir::MLIRContext *context, std::int64_t rank) {
        return GridAttr::get(context, SmallVector<std::int64_t>(rank, 1));
      }

      uint64_t mutable cGridVolume = 0;
      uint64_t getGridVolume() const {
        if (cGridVolume != 0) {
          return cGridVolume;
        }

        cGridVolume = 1;
        for (int64_t dim : getShape()) {
          cGridVolume *= dim;
        }
        return cGridVolume;
      }

      uint64_t getRank() const { return getShape().size(); }
  }];
}

def TTCore_ArchAttr : EnumAttr<TTCore_Dialect, TTCore_Arch, "arch"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_DataTypeAttr : EnumAttr<TTCore_Dialect, TTCore_DataType, "supportedDataTypes"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_CoreCoordAttr : TTCore_Attr<"CoreCoord", "core_coord"> {
  let summary = "TT core_coord attribute";
  let description = [{
    TT core_coord attribute containing a single physical core coordinate.
  }];

  let parameters = (ins "int64_t":$y, "int64_t":$x);
  let assemblyFormat = "custom<VargDimensionList>($y, $x)";
}

def TTCore_TileSizeAttr : TTCore_Attr<"TileSize", "tile_size"> {
  let summary = "TT tile_size attribute";
  let description = [{
    TT tile_size attribute containing a supported Tensix tile shape.
  }];

  let parameters = (ins "int64_t":$y, "int64_t":$x);
  let assemblyFormat = "custom<VargDimensionList>($y, $x)";
}

def TTCore_ChipDescAttr : TTCore_Attr<"ChipDesc", "chip_desc"> {
  let summary = "TT chip_desc attribute";
  let description = [{
    TT chip_desc attribute
  }];

  let parameters = (ins "ArchAttr":$arch,
                    ArrayRefParameter<"int64_t">:$grid,
                    ArrayRefParameter<"int64_t">:$coordTranslationOffsets,
                    "unsigned":$l1Size,
                    "unsigned":$numDramChannels,
                    "unsigned":$dramChannelSize,
                    "unsigned":$nocL1AddressAlignBytes,
                    "unsigned":$pcieAddressAlignBytes,
                    "unsigned":$nocDRAMAddressAlignBytes,
                    "unsigned":$l1UnreservedBase,
                    "unsigned":$eriscL1UnreservedBase,
                    "unsigned":$dramUnreservedBase,
                    "unsigned":$dramUnreservedEnd,
                    ArrayRefParameter<"DataTypeAttr">:$supportedDataTypes,
                    ArrayRefParameter<"TileSizeAttr">:$supportedTileSizes,
                    "unsigned":$dstRegisterSizeTiles,
                    "unsigned":$numCBs,
                    "unsigned":$numComputeThreads,
                    "unsigned":$numDatamovementThreads);
  let assemblyFormat = [{`{` `arch` `=` $arch `,`
                             `grid` `=` custom<DimensionList>($grid) `,`
                             `coord_translation_offsets` `=` custom<DimensionList>($coordTranslationOffsets) `,`
                             `l1_size` `=` $l1Size `,`
                             `num_dram_channels` `=` $numDramChannels `,`
                             `dram_channel_size` `=` $dramChannelSize `,`
                             `noc_l1_address_align_bytes` `=` $nocL1AddressAlignBytes `,`
                             `pcie_address_align_bytes` `=` $pcieAddressAlignBytes `,`
                             `noc_dram_address_align_bytes` `=` $nocDRAMAddressAlignBytes  `,`
                             `l1_unreserved_base` `=` $l1UnreservedBase `,`
                             `erisc_l1_unreserved_base` `=` $eriscL1UnreservedBase `,`
                             `dram_unreserved_base` `=` $dramUnreservedBase `,`
                             `dram_unreserved_end` `=` $dramUnreservedEnd `,`
                             `supported_data_types` `=` `[` $supportedDataTypes `]` `,`
                             `supported_tile_sizes` `=` `[` $supportedTileSizes `]` `,`
                             `dst_register_size_tiles` `=` $dstRegisterSizeTiles `,`
                             `num_cbs` `=` $numCBs `,`
                             `num_compute_threads` `=` $numComputeThreads `,`
                             `num_datamovement_threads` `=` $numDatamovementThreads `}`}];

  let extraClassDeclaration = [{
    unsigned getUsableL1Size() const { return getL1Size() - getL1UnreservedBase(); }
    unsigned getUsableDramChannelSize() const { return getDramUnreservedEnd() - getDramUnreservedBase(); }
  }];
}

def TTCore_CPURoleAttr : EnumAttr<TTCore_Dialect, TTCore_CPURole, "cpu_role"> {
  let assemblyFormat = "$value";
}

def TTCore_CPUDescAttr : TTCore_Attr<"CPUDesc", "cpu_desc"> {
  let summary = "TT cpu_desc attribute";
  let description = [{
    TT cpu_desc attribute
  }];

  let parameters = (ins "CPURole":$role,
                        "StringAttr":$target_triple);
  let assemblyFormat = [{`{` `role` `=` $role `,`
                             `target_triple` `=` $target_triple `}`}];
}

def TTCore_TensorMeshAttr : TTCore_Attr<"TensorMesh", "tensor_mesh", []> {
  let summary = "Tensor mesh in TT dialect.";
  let description = [{
    Describes what mesh a tensor lives on.
  }];
  let parameters = (ins "StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
  let extraClassDeclaration = [{
      static TensorMeshAttr get(::mlir::MLIRContext *context, StringRef name) {
        auto meshNameStrAttr = mlir::StringAttr::get(context, name);
        return TensorMeshAttr::get(context, meshNameStrAttr);
      }
  }];
}

def TTCore_ChipCoordAttr : TTCore_Attr<"ChipCoord", "chip_coord"> {
  let summary = "TT chip_coord attribute";
  let description = [{
    TT chip_coord attribute
  }];

  let parameters = (ins "unsigned":$rack, "unsigned":$shelf, "unsigned":$y, "unsigned":$x);
  let assemblyFormat = "custom<VargDimensionList>($rack, $shelf, $y, $x)";
}

def TTCore_ChipChannelAttr : TTCore_Attr<"ChipChannel", "chip_channel"> {
  let summary = "TT chip_channel attribute";
  let description = [{
    TT chip_channel attribute
  }];

  let parameters = (ins "unsigned":$deviceId0,
                        ArrayRefParameter<"int64_t">:$ethernetCoreCoord0,
                        "unsigned":$deviceId1,
                        ArrayRefParameter<"int64_t">:$ethernetCoreCoord1);
  let assemblyFormat = "`<` `[` $deviceId0 `,` $ethernetCoreCoord0 `]` `,` `[` $deviceId1 `,` $ethernetCoreCoord1 `]` `>`";
}

def TTCore_SystemDescAttr : TTCore_Attr<"SystemDesc", "system_desc"> {
  let summary = "TT system_desc attribute";
  let description = [{
    TT system_desc attribute
  }];

  let parameters = (ins ArrayRefParameter<"CPUDescAttr">:$cpuDescs,
                        ArrayRefParameter<"ChipDescAttr">:$chipDescs,
                        ArrayRefParameter<"unsigned">:$chipDescIndices,
                        ArrayRefParameter<"ChipCapabilityAttr">:$chipCapabilities,
                        ArrayRefParameter<"ChipCoordAttr">:$chipCoords,
                        OptionalArrayRefParameter<"ChipChannelAttr">:$chipChannels);
  let assemblyFormat = "`<` `[` $cpuDescs `]` `,` `[` $chipDescs `]` `,` `[` $chipDescIndices `]` `,` `[` $chipCapabilities `]` `,` `[` $chipCoords `]` (`,` `[` $chipChannels^ `]`)? `>`";

  let extraClassDeclaration = [{
    static SystemDescAttr getDefault(MLIRContext *context, Arch arch = Arch::WormholeB0, const llvm::SmallVector<int64_t> &meshShape = {1});
    static FailureOr<SystemDescAttr> getFromPath(MLIRContext *context, StringRef path, llvm::function_ref<mlir::InFlightDiagnostic()> diagFn);
    static FailureOr<SystemDescAttr> getFromBuffer(MLIRContext *context, void *systemDesc, llvm::function_ref<mlir::InFlightDiagnostic()> diagFn);
    ChipDescAttr getChipDesc(unsigned chipIndex) const;
    unsigned getAddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getAddressAlignBytes(MemorySpace memorySpace, unsigned chipIndex = 0) const;
    unsigned getNocL1AddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getNocDRAMAddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getPcieAddressAlignBytes(unsigned chipIndex = 0) const;
  }];
}

def TTCore_ViewLayoutAttr : TTCore_Attr<"ViewLayout", "view", [TTCore_DeviceLayoutInterface, MemRefLayoutAttrInterface]> {
  let summary = "View layout attribute in TT dialect";
  let description = [{
    Describes a view layout of a memref buffer.
    - AffineMap: Provides affine map indexing into the associated data view.

    Only the view_layout or stream_layout ops should return memref's with this attribute.
    The view layout attribute is necessary for two reasons:
      - It provides a way to reblock the data view into a different shape (via affine map).
        Usually this would be some subblock of the original backing memory to chunk the data
        into smaller pieces.
      - The type itself is a signal to datamovement passes that the memref is a view and
        should be treated as such.
  }];

  let parameters = (ins "AffineMap":$affineMap);

  let assemblyFormat = "`<` custom<IdentityAffineMap>($affineMap) `>`";

  let extraClassDeclaration = [{
      // Compose two view layouts f(g(x)) where f=this and g=other.
      ViewLayoutAttr compose(ViewLayoutAttr g) const;

      bool isView() { return true; }
  }];
}

def TTCore_ShardLayoutAttr : TTCore_Attr<"ShardLayout", "shard", [TTCore_DeviceLayoutInterface, MemRefLayoutAttrInterface]> {
  let summary = "Shard layout attribute in TT dialect";
  let description = [{
    Describes shard layout of a memref buffer.
    - Stride: Stride of each dim in bytes.
    - Buffers: Number of back buffers used for double buffering, I/O latency hiding, etc

    The shard layout attribute is a description of how each shard of a memref is laid out in
    memory. Memref's with this layout type implicitly mean their data is distributed across
    a grid of cores.
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$stride,
                        "uint32_t":$buffers,
                        OptionalParameter<"AffineMap">:$coreVirtualizationMap
                        );

  let assemblyFormat = "`<` custom<DimensionList>($stride) `,` $buffers (`,` $coreVirtualizationMap^)? `>`";

  let extraClassDeclaration = [{
    static ShardLayoutAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, uint64_t elementSize, uint32_t buffers);
    static ShardLayoutAttr get(ArrayRef<int64_t> shape, Type elementType, uint32_t buffers);
    static ShardLayoutAttr get(MemRefType memrefType, uint32_t buffers);
    static ShardLayoutAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> strides, uint32_t buffers);

    bool isView() { return false; }

    AffineMap getAffineMap() const;
  }];
}

def TTCore_InterleavedLayoutAttr : TTCore_Attr<"InterleavedLayout", "interleaved", [TTCore_DeviceLayoutInterface, MemRefLayoutAttrInterface]> {
  let summary = "Interleaved layout attribute in TT dialect";
  let description = [{
    Describes overall layout of an interleaved memref buffer.
    - Stride: Stride of each dim in bytes.
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$stride);

  let assemblyFormat = "`<` custom<DimensionList>($stride) `>`";

  let extraClassDeclaration = [{
    static InterleavedLayoutAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, uint64_t elementSize);
    static InterleavedLayoutAttr get(ArrayRef<int64_t> shape, Type elementType);
    static InterleavedLayoutAttr get(MemRefType memrefType);

    bool isView() { return false; }

    AffineMap getAffineMap() const;
  }];
}

def TTCore_HostLayoutAttr : TTCore_Attr<"HostLayout", "host_layout", [MemRefLayoutAttrInterface]> {
  let summary = "Host-side memref layout attribute with padding support.";
  let description = [{
    Describes a host-side memref layout with a logical shape where each
    dimension may be padded up to a requested alignment. This attribute encodes
    row-major strides and volume of the aligned shape, both computed in the
    bufferization pass to match the corresponding device memref's shape.

    This allows host memrefs to reflect the padded footprint required by the
    device while preserving the original logical shape, enabling I/O for
    unaligned densly-packed shapes (e.g., non-multiples of 32×32 tiles). The
    correct element placement in data copys between padded and unpadded host
    memrefs is ensured by the strided-memcpy of the runtime's `memref.copy`
    implementation.

    - logical_shape: The unpadded logical shape in elements.
    - host_strides: Per-dimension alignment in elements used to compute aligned shapes for stride calculation.
    - host_volume: The (potentially padded) volume in elements.
    - optional(mesh): The mesh that a memref lives on. No mesh indicates single device.
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$logical_shape,
                        ArrayRefParameter<"int64_t">:$host_strides,
                        "int64_t":$host_volume,
                        OptionalParameter<"TensorMeshAttr">:$mesh);

  let assemblyFormat = "`<` `logical_shape` `=` custom<DimensionList>($logical_shape) `,` `host_strides` `=` custom<DimensionList>($host_strides) `,` `host_volume` `=` $host_volume (`,` $mesh^)? `>`";

  let extraClassDeclaration = [{
    LogicalResult getStridesAndOffset(ArrayRef<int64_t> shape,
                                      SmallVectorImpl<int64_t> &strides,
                                      int64_t &offset) const;

    AffineMap getAffineMap() const;

    bool isPadded() const;
  }];
}

def TTCore_MetalLayoutAttr : TTCore_Attr<"MetalLayout", "metal_layout", [TTCore_DeviceLayoutInterface]> {
  let summary = "Tensor layout attribute with explicit physical shape";
  let description = [{
    The tensor layout attribute captures how tensor data is sharded across a grid of devices/cores
    and is laid out in memory.  Note that the presence of this attribute implies that the tensor
    shape includes sharding (i.e. the first half of the tensor shape represents the grid shape).

    Some high level goals:
      - **Logical shapes**: Store the original tensor shape and rank intact and agnostic
        to underlying storage layout.
        Keeping the logical shapes not only makes some graph transformations vastly
        simpler, in particular convs, but it makes the lowered IR much easier to read
        and reason about.  The original tensor shapes leave breadcrumbs that make it
        much easier to map back to the input representation.
      - **Collapsed dims**: We may collapse dimensions during transformation, but it
        is important we capture this information such that it is not lost during tensor
        transformation.  The collapsed_intervals field stores the collapses performed
        during conversion from logical_shape to physical tensor shape.
      - **Padding**: store the desired alignments s.t. padding can be simply encoded;
      dim_alignments field represents alignment along each logical dim during collapse.
      - **Memref translation**: ensure we have all necessary info s.t. we can trivally
        lower a tensor into a memref without any intermediate passes.

    For a logical tensor of shape [H, W] distributed across a grid [GY, GX], the tensor shape would be:
    - Without tiling: [GY, GX, H/GY, W/GX]
    - With tiling: [GY, GX, H/GY/TH, W/GX/TW, TH, TW] where TH,TW are tile dimensions

    This makes the representation 1:1 with memrefs and eliminates the need for shape conversion passes.

    Examples:
    ```mlir
    // Logical 8x300 tensor distributed across 1x2 grid:
    // tensor<1x2x8x150xf32, #tt.metal_layout<logical_shape=8x300, ...>>

    // Logical 1024x1024 tensor distributed across 2x2 grid with 32x32 tiles:
    // tensor<2x2x16x16x!ttcore.tile<32x32xf32>, #tt.metal_layout<logical_shape=1024x1024, ...>>
    ```
  }];

  let parameters = (ins
    // The logical tensor shape before distribution.
    ArrayRefParameter<"int64_t">:$logical_shape,

    // Imposed alignment for each collapse stage in each collapsed interval, interpretation requires its corresponding collapsed_intervals.
    // Default is 1x1x...x1x32x32.
    ArrayRefParameter<"int64_t">:$dim_alignments,

    // Dim collapse info; must be encoded as 2D array of pairs, where each pair represent a half-open range. Negative indexing is supported.
    AttrParameter<"DenseIntElementsAttr", "Intervals of dims to collpase">:$collapsed_intervals,

    // Out of bounds value.
    AttrParameter<"OOBVal", "Out of bounds fill value">:$oob_val,

    // Memory space for the tensor.
    DefaultValuedParameter<"MemorySpace", "MemorySpace::System">:$memory_space,

    DefaultValuedParameter<"TensorMemoryLayout", "TensorMemoryLayout::Sharded">:$memory_layout,

    // Optional index remapping (e.g., view remap or virtual→physical grid mapping).
    // Defaults to empty, which is treated as identity.
    DefaultValuedParameter<"AffineMap", "$_builder.getEmptyAffineMap()">:$indexAffineMap
  );

  let assemblyFormat = "`<` `logical_shape` `=` custom<DimensionList>($logical_shape) `,` `dim_alignments` `=` custom<DimensionList>($dim_alignments) `,` `collapsed_intervals` `=` $collapsed_intervals `,` $oob_val (`,` $memory_space^)? (`,` $memory_layout^)? (`,` `index_map` `=` custom<IdentityAffineMap>($indexAffineMap)^)? `>`";

  let extraClassDeclaration = [{
    static MetalLayoutAttr get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout);

    static MetalLayoutAttr get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout,
                                     DenseIntElementsAttr collapseIntervals);
    static MetalLayoutAttr get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout,
                                     DenseIntElementsAttr collapseIntervals,
                                     ArrayRef<int64_t> dimAlignments
                                     );

    // Convenience getter: same as the 5-arg variant but attaches an explicit
    // index_map to the resulting attribute.
    static MetalLayoutAttr get(::mlir::MLIRContext *context,
                                     ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> deviceGridShape,
                                     OOBVal oobVal, MemorySpace memorySpace,
                                     TensorMemoryLayout memoryLayout,
                                     mlir::AffineMap indexAffineMap
                                     );

    // Derive the physical tensor shape from logical shape and grid.
    llvm::SmallVector<int64_t> getPhysicalShape(ArrayRef<int64_t> tileShape) const;

    static llvm::SmallVector<int64_t> computeAlignments(ArrayRef<int64_t> logicalShape,
                                   ArrayRef<int64_t> deviceGridShape,
                                   ArrayRef<int64_t> normIntervals);

    llvm::SmallVector<int64_t> getDeviceShape(ArrayRef<int64_t> gridShape, ArrayRef<int64_t> tileShape) const;

    // Get the corresponding memref type (essentially just drops grid dimensions).
    static MemRefType getMemRefType(RankedTensorType tensorType);

    // Stride calculations for shard portion.
    llvm::SmallVector<int64_t> getShardStride(RankedTensorType tensorType) const;

    // Calculate and return the aligned host tensor strides and volume.
    std::pair<llvm::SmallVector<int64_t>, int64_t> getHostStrideAndVolume() const;
    llvm::SmallVector<int64_t> getHostStride() const;
    int64_t getHostVolume() const;

    llvm::SmallVector<int64_t> getNormalizedIntervals() const;

    // Returns the explicit index_map if present; otherwise returns an
    // identity map of the provided rank.
    mlir::AffineMap getIndexAffineMapOrIdentity(unsigned rank) const;

    bool isView() { return false; }
  }];
}

def TTCore_DeviceAttr : TTCore_Attr<"Device", "device", []> {
  let summary = "Device attribute in TT dialect.";
  let description = [{
    Describes the physical layout of a device in the system and is made up of a few components:
    - A grid attribute that describes the device's compute grid shape.  It not only describes the shape of the compute grid, but also
      carries an affine map that describes how the logical grid maps to the physical grid.
    - Two affine maps that describe how a tensor layout's linear attribute maps to the L1 and DRAM memory spaces.
    - A mesh shape that describes the virtual layout of the chips with respect to each other. Note that in a multi-chip system, this grid
      encapsulates the entire system's grid shape, e.g. 8x16 grid could be made up of a 1x2 mesh of chips side-by-side. The mesh
      attribute configures how the above grid/map attributes are created such that they implement this mesh topology.
    - An array of chip ids that this device is made up of. This array's length must match the volume of the mesh shape and should be
      interpreted in row-major order.
  }];
  let parameters = (ins TTCore_GridAttr:$workerGrid,
                        "AffineMap":$l1Map,
                        "AffineMap":$dramMap,
                        ArrayRefParameter<"int64_t">:$meshShape,
                        ArrayRefParameter<"unsigned">:$chipIds);
  let assemblyFormat = "`<` `workerGrid` `=` qualified($workerGrid) `,` `l1Map` `=` qualified($l1Map) `,` `dramMap` `=` qualified($dramMap) `,` `meshShape` `=` custom<DimensionList>($meshShape) `,` `chipIds` `=` `[` $chipIds `]` `>`";

  let extraClassDeclaration = [{
      static DeviceAttr get(::mlir::MLIRContext *context, SystemDescAttr systemDesc, ArrayRef<int64_t> meshShape, ArrayRef<unsigned> chipIds);
      static DeviceAttr get(::mlir::MLIRContext *context, SystemDescAttr systemDesc, ArrayRef<int64_t> meshShape = {});
      AffineMap getMemoryMap(MemRefType memrefType,
                             size_t pageSize,
                             std::optional<AffineMap> view = std::nullopt,
                             size_t baseOffset = 0) const;
      AffineMap getMemoryMap(std::pair<MemRefType, AffineMap> memrefAndView,
                             size_t pageSize,
                             size_t baseOffset = 0) const;

      // Returns the size in bytes of the given memref type. Supports memref's of multiple flavors:
      //   - Generic region memrefs, e.g. memref<64x128xf32, #l1_> this will return the size of the whole memref.
      //   - Device layout memrefs, e.g. memref<1x2x3x4x!tt.tile<32x32, f32>, #l1_> this will return the size of
      //     the shard, just 3*4*sizeof(!ttcore.tile).
      //
      // Arguments:
      //   - pageSize: Effectively the alignment size of the memref's size. If 0, will use the default alignment
      //     of the memref type for the given memory space.
      //   - includeBuffers: If true, will multiply the size by the number of back buffers.
      size_t getMemrefSizeBytes(MemRefType memrefType, size_t pageSize = 0, bool includeBuffers = false) const;

      // Returns the size in bytes of a single shard of the given memref type.
      //
      // This function computes the size of a single shard for a memref that may be sharded across a device grid.
      // If the memref uses a ShardLayout, the shard shape is determined from the ShardLayout; otherwise, the memref's
      // shape is used directly. The size is calculated as the product of the shard shape dimensions and the element
      // size in bytes, optionally multiplied by the number of buffers if includeBuffers is true. The result is then
      // aligned up to the specified alignSize.
      //
      // Arguments:
      //   - memrefType: The memref type whose shard size is to be computed.
      //   - alignSize: The alignment size in bytes to which the result will be rounded up.
      //   - includeBuffers: If true and the layout specifies multiple buffers, the size will be multiplied accordingly.
      //
      size_t getShardSizeInBytes(MemRefType memrefType, size_t alignSize, bool includeBuffers) const;

      // Returns the circular buffer page size in bytes of the given memref type. The general heursitic is:
      //   - If element type is tile type, return the size of the tile type.
      //   - If element type is scalar type, return the size of a tile type with underlying scalar type.
      size_t getMemrefCBPageSizeBytes(MemRefType memrefType) const;

      // Returns the memref size bytes divided by the circular buffer page size.
      size_t getMemrefCBNumPages(MemRefType memrefType) const;

      // Returns the footprint size in bytes of the tensor layout distributed across the given memory space.
      // The resulting size is a function of the memory space, roughly speaking this ends up being:
      // - DeviceL1: This ends up being exactly the shard size
      // - DeviceDRAM: Is more nuanced because the whole tensor size gets paged and interleaved between all dram channels,
      //   due to paging and rounding the footprint ends up being close to: the_whole_tensor / num_dram_channels
      uint64_t getLayoutSizeBytes(ArrayRef<int64_t> tensorShape, MetalLayoutAttr layout, MemorySpace memorySpace) const;

      // Returns the footprint size in bytes of the tensor distributed across the given memory space.
      // Forwards to getLayoutSizeBytes, see comment there for more info.
      uint64_t getTensorSizeBytes(RankedTensorType tensorType, MemorySpace memorySpace) const;
  }];

  let genVerifyDecl = 1;
}

def TTCore_MemorySpaceAttr : EnumAttr<TTCore_Dialect, TTCore_MemorySpace, "memory_space"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_OOBValAttr : EnumAttr<TTCore_Dialect, TTCore_OOBVal, "oob_val"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_IteratorTypeAttr : EnumAttr<TTCore_Dialect, TTCore_IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_IteratorTypeArrayAttr : TypedArrayAttrBase<TTCore_IteratorTypeAttr, "">;

def TTCore_ArgumentAllocationAttr : TTCore_Attr<"ArgumentAllocation", "arg_alloc", []> {
  let summary = "Argument allocation attribute in TT dialect";
  let description = [{
    Holds the metadata for the allocation of an function argument i.e. for graph inputs.
  }];
  let parameters = (ins "uint64_t":$address, "uint64_t":$size, "MemorySpace":$memorySpace);
  let assemblyFormat = "`<` $address `,` $size `,` $memorySpace `>`";
}

def TTCore_ReduceTypeAttr : EnumAttr<TTCore_Dialect, TTCore_ReduceType, "reduce_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_ReduceTypeArrayAttr : TypedArrayAttrBase<TTCore_ReduceTypeAttr, "">;

def TTCore_MeshShardDirectionAttr : EnumAttr<TTCore_Dialect, TTCore_MeshShardDirection, "shard_direction"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_ShardStatusAttr : EnumAttr<TTCore_Dialect, TTCore_ShardStatus, "shard_status"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_MeshShardTypeAttr : EnumAttr<TTCore_Dialect, TTCore_MeshShardType, "shard_type"> {
  let summary = "MeshShard shard_type attribute in TT dialect";
  let description = [{
    Define sharded tensor data of mesh_shard op.
    - Identity: input and output tensors are pre-sharded (same data) and no sharding is required.
    - Replicate: all of the devices has full tensor (same data).
    - Maximal: one or part of the devcices has full tensor (same data).
    - Devices: all or part of the devices has sharded (partial) tensor (different data).
  }];
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_MeshAttr : TTCore_Attr<"Mesh", "mesh", []> {
  let summary = "Mesh reference attribute in TT dialect.";
  let description = [{
    Describes a mesh config including name and shape.
  }];
  let parameters = (ins "StringAttr":$name,
                        ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` $name `=` custom<DimensionList>($shape) `>`";
}

def TTCore_MeshesAttr : TTCore_Attr<"Meshes", "meshes"> {
  let summary = "TT system meshes attribute.";
  let description = [{
    TT system meshes attribute includes one or more mesh configs used for networks.
  }];
  let parameters = (ins ArrayRefParameter<"MeshAttr">:$meshes);
  let assemblyFormat = "`<` `[` $meshes `]` `>`";
  let extraClassDeclaration = [{
      MeshAttr getMesh(StringRef name) {
        for( auto mesh : getMeshes() ) {
          if( mesh.getName() == name ) {
            return mesh;
          }
        }
        return nullptr;
      }
  }];
}

//===----------------------------------------------------------------------===//
// TT type definitions
//===----------------------------------------------------------------------===//

class TTCore_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTCore_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTCore_Tile : TTCore_Type<"Tile", "tile", [MemRefElementTypeInterface, FloatTypeInterface]> {
    let summary = "TT tile";
    let description = "Tile type in TT dialect";
    let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "DataType":$dataType);
    let assemblyFormat = "`<` custom<DimensionList>($shape) `,` $dataType `>`";

    let extraClassDeclaration = [{
      static constexpr std::array<int64_t, 2> getDefaultShape() { return {32, 32}; }
      static TileType get(Type elementType, ArrayRef<int64_t> shape = getDefaultShape());
      SmallVector<int64_t> getScalarShape(SmallVector<int64_t> tiledShape) const;
      SmallVector<int64_t> getTiledShape(SmallVector<int64_t> scalarShape) const;
      uint64_t getSizeBytes() const;
      int64_t getHeight() const { return getShape()[0]; }
      int64_t getWidth() const { return getShape()[1]; }
      // Returns the scalar element type of the tile, if compressed it returns
      // the corresponding uncompressed element type, i.e. bfp_bf8 -> bf16
      Type getElementType() const;
      const ::llvm::fltSemantics &getFloatSemantics() {
        llvm_unreachable("TTCore_Tile::getFloatSemantics unimplemented #5124");
      }
    }];

    let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Auxiliary type definitions
//===----------------------------------------------------------------------===//

def TTCore_Tuple : NestedTupleOf<[AnyRankedTensor]>;

def TTCore_TupleMemberType : AnyTypeOf<[AnyRankedTensor]>;

def TTCore_ArgumentTypeAttr : EnumAttr<TTCore_Dialect, TTCore_ArgumentType, "argument_type"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // TTMLIR_DIALECT_TTCore_IR_TTCOREOPSTYPES_TD
