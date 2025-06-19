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


def TTCore_ChipPhysicalHelperCoresAttr : TTCore_Attr<"ChipPhysicalHelperCores", "chip_physical_helper_cores"> {
  let summary = "TT chip_physical_helper_cores attribute";
  let description = [{
    TT chip_physical_helper_cores attribute containing arrays of physical helper cores by core type in order of logical cores.
  }];

  let parameters = (ins ArrayRefParameter<"CoreCoordAttr">:$dram, OptionalArrayRefParameter<"CoreCoordAttr">:$eth, OptionalArrayRefParameter<"CoreCoordAttr">:$eth_inactive);
  let assemblyFormat = "`{` `dram` `=` `[` $dram `]` (`eth` `=` `[` $eth^ `]`)? (`eth_inactive` `=` `[` $eth_inactive^ `]`)? `}`";
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
                    "ChipPhysicalHelperCoresAttr":$chipPhysicalHelperCores,
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
                             `physical_helper_cores` `=` $chipPhysicalHelperCores `,`
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
    static tt::SystemDescAttr getDefault(MLIRContext *context, tt::Arch arch = tt::Arch::WormholeB0, const llvm::SmallVector<int64_t> &meshShape = {1});
    static FailureOr<tt::SystemDescAttr> getFromPath(MLIRContext *context, StringRef path, llvm::function_ref<mlir::InFlightDiagnostic()> diagFn);
    ChipDescAttr getChipDesc(unsigned chipIndex) const;
    unsigned getAddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getAddressAlignBytes(MemorySpace memorySpace, unsigned chipIndex = 0) const;
    unsigned getNocL1AddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getNocDRAMAddressAlignBytes(unsigned chipIndex = 0) const;
    unsigned getPcieAddressAlignBytes(unsigned chipIndex = 0) const;
  }];
}

def TTCore_ViewLayoutAttr : TTCore_Attr<"ViewLayout", "view", [TTCore_DeviceLayoutInterface]> {
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
      static ViewLayoutAttr get(::mlir::MLIRContext *context, unsigned rank) {
        return get(context, mlir::AffineMap::getMultiDimIdentityMap(rank, context));
      }

      // Compose two view layouts f(g(x)) where f=this and g=other.
      ViewLayoutAttr compose(ViewLayoutAttr g) const;
  }];
}

def TTCore_ShardLayoutAttr : TTCore_Attr<"ShardLayout", "shard", [TTCore_DeviceLayoutInterface]> {
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
                        DefaultValuedParameter<"uint32_t", "1">:$buffers);

  let assemblyFormat = "`<` custom<DimensionList>($stride) (`,` $buffers^)? `>`";

  let extraClassDeclaration = [{
    static ShardLayoutAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, uint64_t elementSize, uint32_t buffers);
    static ShardLayoutAttr get(ArrayRef<int64_t> shape, Type elementType, uint32_t buffers);
    static ShardLayoutAttr get(MemRefType memrefType, uint32_t buffers);

    AffineMap getAffineMap() const;
  }];
}

def TTCore_MetalLayoutAttr : TTCore_Attr<"MetalLayout", "metal_layout"> {
  let summary = "Tensor layout attribute";
  let description = [{
    The tensor layout attribute captures how tensor data is sharded across a grid of devices, cores, and
    is laid out in memory.

    Some high level goals
      - **Logical shapes**: Keep the original tensor shape and rank intact and agnostic
        to underlying storage layout.
        Keeping the logical shapes not only makes some graph transformations vastly
        simpler, in particular convs, but it makes the lowered IR much easier to read
        and reason about.  The original tensor shapes leave breadcrumbs that make it
        much easier to map back to the input representation.
      - **Flexible sharding**: Enable flexibility in choosing grid shape, to get better
        parallelization and avoid resharding. This is particularly important in cases
        where tensor shapes are not clean powers of two and would otherwise force our
        hand in choosing non-optimal grid shapes.
      - **Logical-Physical Isomorphism**: Encode this information with just a few
        attributes to enable derived conversions from logical to physical layout and back.
      - **Explicit**: A single source of truth.
      - Enable a direct way to query padded regions.

    Please refer to the [Tensor Layout Spec](https://tenstorrent.github.io/tt-mlir/specs/tensor-layout.html) for more in depth documentation.

    Examples:
    ```mlir
    tensor<8x300xf32,
      #ttcore.metal_layout<(d0, d1) -> (d0, d1),
        undef,
        <1x2>,
        memref<8x150xf32, #ttcore.memory_space<l1>>
      >
    >

    tensor<8x96x32xf32,
      #ttcore.metal_layout<(d0, d1, d2) -> (d0 * 96 + d1, d2),
        undef,
        <2x1>,
        memref<384x32xf32, #ttcore.memory_space<l1>>
      >
    >

    tensor<8x96x32xf32,
      #ttcore.metal_layout<(d0, d1, d2) -> (d0 * 96 + d1, d1, d2),
        undef,
        <2x1x2>,
        memref<384x96x16xf32, #ttcore.memory_space<l1>>
      >
    >

    tensor<5x3x2x2x7x32x32xf32,
      #ttcore.metal_layout<
        (d0, d1, d2, d3, d4, d5, d6)
          -> (d0 * 2688 + d1 * 896 + d2 * 448 + d3 * 224 + d4 * 32 + d5, d4, d5, d6),
        undef,
        <3x2x2x2>,
        memref<4480x4x16x16xf32, #ttcore.memory_space<l1>>
      >
    >
    ```
  }];

  let parameters = (ins AttrParameter<"AffineMap", "An affine map that defines how the logical tensor dimensions map to a grid shape.">:$linear,
                        AttrParameter<"OOBVal", "A tracked out of bounds value that fills padding space.">:$oob_val,
                        AttrParameter<"GridAttr", "The grid shape that this tensor is divided onto.">:$grid,
                        AttrParameter<"MemRefType", "A memref that describes the physical footprint allocation of the shard. It must also have a shape with rank equal to grid.">:$memref);
  let assemblyFormat = "`<` $linear`,` $oob_val`,` $grid`,` $memref `>`";

  let extraClassDeclaration = [{
      static MetalLayoutAttr get(::mlir::MLIRContext *context,
                            RankedTensorType ty,
                            uint64_t gridRank,
                            bool tiled = false,
                            MemorySpace memorySpace = MemorySpace::System,
                            ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                            OOBVal oobVal = OOBVal::Undef);
      static MetalLayoutAttr get(::mlir::MLIRContext *context,
                            RankedTensorType ty,
                            GridAttr grid,
                            bool tiled = false,
                            MemorySpace memorySpace = MemorySpace::System,
                            ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                            OOBVal oobVal = OOBVal::Undef);
      static MetalLayoutAttr get(::mlir::MLIRContext *context,
                            ArrayRef<int64_t> tensorShape,
                            Type elementType,
                            MemorySpace memorySpace = MemorySpace::System,
                            GridAttr grid = {},
                            ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                            OOBVal oobVal = OOBVal::Undef);
      static MetalLayoutAttr get(::mlir::MLIRContext *context,
                            RankedTensorType ty,
                            MemorySpace memorySpace = MemorySpace::System,
                            GridAttr grid = {},
                            ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}},
                            OOBVal oobVal = OOBVal::Undef);
      static MetalLayoutAttr get(::mlir::MLIRContext *context,
                            RankedTensorType ty,
                            MemorySpace memorySpace,
                            GridAttr grid,
                            Type elementType);
      MetalLayoutAttr withGrid(::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid, ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
      MetalLayoutAttr withGrid(::mlir::MLIRContext *context,
                          RankedTensorType ty,
                          GridAttr grid,
                          ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals = {{0, -1}});
      MetalLayoutAttr withElementType(::mlir::MLIRContext *context, Type elementType);
      MetalLayoutAttr withMemorySpace(::mlir::MLIRContext *context, MemorySpace memorySpace);
      MetalLayoutAttr withShardShape(::mlir::MLIRContext *context, llvm::SmallVector<int64_t> shardShape);

      uint64_t getMemrefSizeBytes() const;
      MemorySpace getMemorySpace() const;
      bool isSystemMemorySpace() const { return ::mlir::tt::isSystemMemorySpace(getMemorySpace()); }
      bool isDeviceMemorySpace() const { return ::mlir::tt::isDeviceMemorySpace(getMemorySpace()); }
      bool isTiled() const;
      Type getElementType() const;
      Type getScalarElementType() const;
      uint64_t getElementSizeBytes() const;
      llvm::SmallVector<int64_t> getStride(ArrayRef<int64_t> logicalShape) const;
      llvm::SmallVector<int64_t> getPhysicalShape(ArrayRef<int64_t> logicalShape) const;
      llvm::SmallVector<int64_t> getShardShape(bool convertTileToScalar = true) const;
      llvm::SmallVector<int64_t> getShardStride() const;
      uint64_t getRank() const { return getLinear().getNumResults(); }
      AffineMap replaceMemoryMapSymbolsWithShardShape(AffineMap physicalMemoryMap) const;
      AffineMap projectOnto(AffineMap linearMap, AffineMap physicalMemoryMap) const;
      AffineMap getIdentityTileLinearMap() const;
      llvm::SmallVector<int64_t> getTiledShape(ArrayRef<int64_t> logicalTensorShape) const;
      // Concatenates the grid shape with the memref shape to form an all in one memref shape,
      // used as the buffer type for bufferization, e.g.:
      //   metal_layout<..., <2x3>, memref<4x5xttcore.tile<32x32, f32>>>
      //     -> memref<2x3x4x5xttcore.tile<32x32, f32>>
      MemRefType getBufferType(bool isView = false) const;
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

def TTCore_TensorMeshShardingAxisAttr : TTCore_Attr<"TensorMeshShardingAxis", "tensor_sharding", []> {
  let summary = "Tensor mesh sharding axis info attribute in TT dialect.";
  let description = [{
    Details per tensor dimension sharding and axes info.
    - shard_shape: shard shape at a tensor dimension.
    - (optional) axes: mesh shard dimensions. Axes may be empty if it is not being sharded.
  }];
  let parameters = (ins "int64_t":$shard_shape,
                        OptionalArrayRefParameter<"int64_t">:$axes);
  let assemblyFormat = "$shard_shape (`(`$axes^`)`)?";
}

def TTCore_TensorMeshShardingAttr : TTCore_Attr<"TensorMeshSharding", "mesh_sharding", []> {
  let summary = "Tensor mesh sharding attribute in TT dialect.";
  let description = [{
    Describes a tensor's multi-device status.
    - Single device tensor has no TensorMeshShardingAttr.
        tensor<784x16384xf32>

    - Multi-device tensors have TensorMeshShardingAttr.
      (i) multi-device tensor without tensor mesh shard axis indicates all devices in "mesh"
          have full size tensors e.g., 784x16384 for
            tensor<784x16384xf32, #ttcore.mesh_sharding<"mesh">>

      (ii) multi-device tensor with tensor mesh shard axis indicate all devices in "mesh"
          have sharded tensor defined by the TensorMeshShardingAxisAttr. e.g., 192x16384 for
            tensor<784x16384xf32, #ttcore.mesh_sharding<"mesh", [ 4(1),  1]>>.
          Here, 4(1) indicates shard_shape(shard_dim), so 784 should be sharded by 4
          at "mesh"'s second hardware dimension. 1 indicates no sharding, so 16384 is not
          being sharded.
  }];
  let parameters = (ins "StringAttr":$name,
                        OptionalArrayRefParameter<"TensorMeshShardingAxisAttr">:$tensor_mesh_sharding_axis);
  let assemblyFormat = "`<` $name (`,` `[` $tensor_mesh_sharding_axis^ `]` )? `>`";

  let extraClassDeclaration = [{
      static TensorMeshShardingAttr get(::mlir::MLIRContext *context, StringRef name) {
        auto meshNameStrAttr = mlir::StringAttr::get(context, name);
        ::llvm::SmallVector<TensorMeshShardingAxisAttr> tensor_mesh_sharding_axis;
        return TensorMeshShardingAttr::get(context, meshNameStrAttr, tensor_mesh_sharding_axis);
      }
  }];
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

def TTCore_Tile : TTCore_Type<"Tile", "tile", [MemRefElementTypeInterface]> {
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
