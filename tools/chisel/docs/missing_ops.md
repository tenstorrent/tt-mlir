# Missing TTNN Chisel Golden Wrappers

Ops listed here are registered in `CHISEL_GOLDEN_MAPPINGS` with `raise NotImplementedError`
stubs, or are permanently skipped. They need further investigation before a real
golden wrapper can be written.

## Stubs — registered but not yet implemented

### Ops with no golden function

| TTNN Op | Reason |
|---------|--------|
| `RotaryEmbeddingOp` | No golden function exists in mapping.py |
| `RotaryEmbeddingLlamaOp` | No golden function exists in mapping.py |
| `NLPConcatHeadsOp` | No golden function exists |
| `NLPConcatHeadsDecodeOp` | No golden function exists |
| `MeshShardOp` | No golden function exists |
| `MeshPartitionOp` | No golden function exists |
| `BitcastConvertOp` | `view`-based bitcast is not straightforward for `GoldenMapTensor` |

## Infrastructure ops — skip permanently (no golden needed)

These ops perform device management, I/O, or memory lifecycle with no mathematical
output semantics. Do not add `CHISEL_GOLDEN_MAPPINGS` entries.

`AllocOp`, `DeallocateOp`, `EmptyOp`, `GetDeviceOp`, `D2MSubgraphOp`,
`PointToPointOp`, `BeginTraceCaptureOp`, `EndTraceCaptureOp`, `ExecuteTraceOp`,
`CaptureOrExecuteTraceOp`, `CreateGlobalSemaphoreOp`, `ResetGlobalSemaphoreOp`,
`DumpTensorOp`, `LoadTensorOp`, `WriteTensorOp`, `GenericOp`,
`PrepareConv2dBiasOp`, `PrepareConv2dWeightsOp`,
`PrepareConvTranspose2dBiasOp`, `PrepareConvTranspose2dWeightsOp`
