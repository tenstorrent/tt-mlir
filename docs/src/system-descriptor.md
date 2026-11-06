# System descriptor field guide

A system descriptor records CPU targets, chip resources, and chip connectivity.
The compiler represents it as `#ttcore.system_desc`; a standalone `.ttsys` file
stores a `SystemDescRoot` FlatBuffer containing a `SystemDesc`. For obtaining or
inspecting a descriptor, see [TTRT query](./ttrt.md#query) and
[TTRT read](./ttrt.md#read). The [device attribute](./specs/device.md) describes
a compiler view over this system, including a mesh shape and selected chips.

The tables below use the FlatBuffer field names. In textual MLIR, `grid_size`
is named `grid` and `dram_grid_size` is named `dram_grid`. Two-dimensional
sizes and coordinates are ordered **y, x** (rows, columns for sizes). Memory
sizes, addresses, and alignment values are in **bytes**, unless stated otherwise.

## SystemDesc

| Field | Meaning |
| --- | --- |
| `cpu_descs` | CPU target descriptions. Each entry contains `role` (`Host` or `Device`) and an LLVM `target_triple` string. The runtime producer currently emits one `Host` entry using its build's `TARGET_TRIPLE`. |
| `chip_descs` | Chip resource descriptions; see [ChipDesc](#chipdesc). This array is addressed through `chip_desc_indices`. |
| `chip_desc_indices` | Maps a chip's position in the system descriptor to an index in `chip_descs`. These values are descriptor indices, not TT-Metal device IDs. |
| `chip_capabilities` | Per-chip capability bit masks. `HostMMIO` indicates host MMIO capability; the runtime sets it from `is_mmio_capable()`, otherwise it emits no capability bits. |
| `chip_coords` | Chip placement records with fields `rack`, `shelf`, `y`, and `x`. The current runtime emits only one placeholder `(0, 0, 0, 0)`, even for multiple chips. Do not interpret that output as a measured topology or assume one coordinate record per chip. |
| `chip_channels` | Ethernet connections, each described by two device IDs and their Ethernet core coordinates; see [ChipChannel](#chipchannel). This list is optional in the MLIR assembly format. |

In textual MLIR, the lists in `#ttcore.system_desc` are positional, in the
order shown above; the final channel list may be omitted.

### Chip indices versus device IDs

For a chip position `i`, `SystemDescAttr::getChipDesc(i)` performs:

```text
chip_descs[chip_desc_indices[i]]
```

Both array accesses must be in range. The indirection permits chips to share
a descriptor; it does not require producers to deduplicate descriptors.
The current runtime sorts the queried devices by TT-Metal device ID, appends
one `ChipDesc` per device, and writes `chip_desc_indices = [0, 1, ...]`.

For example, if the queried devices have IDs 4 and 7, their chip positions are
0 and 1, and the runtime emits descriptor indices `[0, 1]`, not `[4, 7]`.
Looking up the second chip uses `getChipDesc(1)`. Ethernet channel endpoints
still use device IDs 4 and 7. A hypothetical deduplicating producer could instead
store one shared descriptor with indices `[0, 0]`; this illustrates the mapping,
not the current runtime output.

The `chipIds` in the compiler's [device attribute](./specs/device.md#examples)
refer to chip indices in the system descriptor. Do not substitute channel
endpoint IDs for these indices. The distinction motivated the correction
discussed in [PR #3758](https://github.com/tenstorrent/tt-mlir/pull/3758#issuecomment-2968074168).

## ChipDesc

| Field | Meaning and current runtime source |
| --- | --- |
| `arch` | Chip architecture: `Wormhole_b0`, `Blackhole`, or `Quasar` in the FlatBuffer schema, obtained from `device->arch()`. |
| `grid_size` (`grid` in MLIR) | Compute-with-storage worker grid dimensions, from `compute_with_storage_grid_size()`. This is a worker grid, not the mesh shape of multiple chips or a list of every physical core on the chip. |
| `coord_translation_offsets` | Translated NoC0 coordinate of logical worker `(0, 0)`: the producer calls `worker_core_from_logical_core({0, 0})`, then `virtual_noc0_coordinate(0, workerNWCorner)`, and stores the result in y, x order. This pair is not a full per-core coordinate mapping. |
| `l1_size` | L1 capacity per core, from `l1_size_per_core()`. It includes the region below `l1_unreserved_base`. |
| `num_dram_channels` | Number of DRAM channels, from `num_dram_channels()`. |
| `dram_channel_size` | DRAM capacity per channel, from `dram_size_per_channel()`, rather than total chip DRAM capacity. |
| `noc_l1_address_align_bytes` | NoC L1 address alignment, from HAL `get_l1_alignment()`. |
| `pcie_address_align_bytes` | PCIe address alignment, from HAL `get_pcie_alignment()`. |
| `noc_dram_address_align_bytes` | NoC DRAM address alignment, from HAL `get_dram_alignment()`. |
| `l1_unreserved_base` | Start of the L1 region available to the allocator, from `get_base_allocator_addr(HalMemType::L1)`. `getUsableL1Size()` subtracts this address from `l1_size`. |
| `erisc_l1_unreserved_base` | Unreserved Ethernet RISC L1 base, from HAL `get_erisc_l1_unreserved_base()`. This is separate from the worker L1 base. |
| `dram_unreserved_base` | Start of the allocator's unreserved DRAM region, from `get_base_allocator_addr(HalMemType::DRAM)`. |
| `dram_unreserved_end` | Exclusive upper bound of the compiler-usable DRAM region per channel. `calculateDRAMUnreservedEnd()` conservatively reserves space for kernel programs at the top of DRAM and rounds the remaining end address down to the allocator's DRAM alignment. This is a calculated bound, not a free-memory measurement. |
| `supported_data_types` | Data types advertised by the descriptor. The runtime currently uses explicit lists, including a separate Quasar list, rather than querying support for each type. This is not a promise that every operation supports every listed type. |
| `supported_tile_sizes` | Advertised Tensix tile shapes in y, x order. The runtime currently uses the fixed list `(4, 16)`, `(16, 16)`, `(32, 16)`, `(4, 32)`, `(16, 32)`, `(32, 32)`. |
| `dst_physical_size_tiles` | Physical destination-register capacity expressed in tiles; currently set to 16 by the runtime producer. The compiler's logical tile capacity can be smaller: `getDstLogicalSizeTiles()` halves it when full synchronization is disabled and halves it again for 32-bit data types. |
| `num_cbs` | Number of circular buffers, taken from TT-Metal's `NUM_CIRCULAR_BUFFERS` constant. |
| `num_compute_threads` | Compute-thread count recorded for a worker. The runtime sets 1 for Wormhole/Blackhole and 4 for Quasar. |
| `num_datamovement_threads` | Programmable data-movement-thread count recorded for a worker. The runtime sets 2 for Wormhole/Blackhole and 6 for Quasar; its Quasar comment distinguishes 6 programmable from 8 physical DM cores. |
| `dram_grid_size` (`dram_grid` in MLIR) | DRAM grid dimensions from `dram_grid_size()`, in y, x order. |
| `dram_bank_to_logical_worker_noc0` | DRAM-bank-indexed logical worker coordinates returned by `get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)`. The runtime skips this query for Quasar and emits an empty list. |
| `dram_bank_to_logical_worker_noc1` | Corresponding bank-to-logical-worker assignment for `NOC_1`; also empty in the current Quasar runtime output. |

The compiler computes usable DRAM bytes per channel as
`dram_unreserved_end - dram_unreserved_base`. Neither this value nor
`l1_size - l1_unreserved_base` reports how much memory is currently free after
allocations.

## ChipChannel

| Field | Meaning |
| --- | --- |
| `device_id0` | TT-Metal device ID of the first endpoint. |
| `ethernet_core_coord0` | First endpoint's Ethernet core coordinate, serialized in y, x order. |
| `device_id1` | TT-Metal device ID of the second endpoint. |
| `ethernet_core_coord1` | Second endpoint's Ethernet core coordinate, serialized in y, x order. |

The runtime's `getAllDeviceConnections()` uses active Ethernet cores and
`get_connected_ethernet_core()`. It orders endpoints by device ID and deduplicates
the resulting connections. Connections whose lookup throws are skipped, so the
list should not be treated as a complete inventory of all external connections.
The coordinates come from those TT-Metal Ethernet APIs; they are not worker-grid
indices or chip placement coordinates.

## SystemDescRoot (.ttsys envelope)

| Field | Meaning |
| --- | --- |
| `version` | TT-MLIR producer version, with `major`, `minor`, and `patch` components. |
| `schema_hash` | Generated system-descriptor FlatBuffer schema hash. |
| `ttmlir_git_hash` | Producer's TT-MLIR Git revision string. |
| `product_identifier` | Product identifier string; the current runtime writes the placeholder `"unknown"`. |
| `system_desc` | The `SystemDesc` payload described above. |

The standalone format is a size-prefixed FlatBuffer with file identifier `TTSY`
and extension `.ttsys`. The envelope metadata is separate from the fields of
the compiler's `#ttcore.system_desc` attribute.

## Queried descriptors and compiler defaults

A runtime query obtains resource values from the opened mesh's devices, HAL,
and allocators, but it also includes the fixed values and placeholders noted
above. In particular, `chip_coords` and `product_identifier` do not identify
the actual physical installation.

`SystemDescAttr::getDefault()` instead constructs architecture-specific defaults
in the compiler. These are not a hardware query. For multiple chips, the current
default builders assume a ring topology with `(0, 0)` Ethernet core coordinates.
Use the producer and provenance of a descriptor when interpreting its values;
do not treat default or placeholder topology as discovered connectivity.

## Source reference

The field definitions and the producer/consumer behavior described here are in:

- [FlatBuffer resource types](https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Target/Common/types.fbs) and [standalone root](https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Target/Common/system_desc.fbs).
- [TTCore attribute definitions and assembly formats](https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td).
- [Runtime descriptor producer](https://github.com/tenstorrent/tt-mlir/blob/main/runtime/lib/common/system_desc.cpp), including `getCurrentSystemDescImpl()`, `getCoordinateTranslationOffsets()`, `calculateDRAMUnreservedEnd()`, and `getAllDeviceConnections()`.
- [Compiler defaults, FlatBuffer reader, and chip lookup](https://github.com/tenstorrent/tt-mlir/blob/main/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp).
- [MLIR-to-FlatBuffer serialization](https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Target/Utils/MLIRToFlatbuffer.h).
