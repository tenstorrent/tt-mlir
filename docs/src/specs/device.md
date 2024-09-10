# Device

Device in tt-mlir is somewhat of an overloaded term and can refer to different
things depending on the context. This document will only speak to the compiler's
abstract representation of a device captured by attribute `#tt.device`.

## Terms

There are many overloaded terms when talking about devices and grids, this
document will use the following definitions:

- **Physical Grid**: A 2D array of tensix cores on a chip.
- **Chip**: A single physical chip with a **Physical Grid** of cores.
- **Card**: A PCIE or Ethernet card that may contain multiple **Chips**.
- **System**: A collection of **Cards** that are usually connected together on the
  same host via PCIE or networked via ethernet.  A system is represented by
  `SystemDesc` in the compiler.
- **Device**: Device is always presented as a single entity to the enclosing
  scope, but it may be virtualized to abstract a multi-card **System** and
  part of its encoding carries a **Logical Grid**. Another way to think of device
  is a view over the system.
- **Logical Grid** or just **Grid**: Is a logical shape that abstracts one or
  more **Physical Grids**.

## Motivation

The device attribute strives to achieve the following goals:
- Provide a convenient representation of a physical grid that decouples the
  logical division of tensors from the physical layout of the hardware. This not
  only simplifies reasoning about how tensors get divided into shards, but can also
  enable reinterpretations of the device grid for data layout optimization decoupled
  from the existing encoding of the tensor layouts.
- Following the first point, the device attribute should be able to represent
  many different forms of logical grids, from simple 2D grids, to more complex
  topologies like extra-wide grids or higher dimensional grids.
- Device attribute captures encoding both single chip and multi-chip systems
  under a single, virtualized representation.
- Enable many forms of data parallel execution strategies for single and
  multi chip systems under a single representation.

## Examples

All of the following examples will assume the physical hardware has an 8x8 physical
grid of cores.  We will use notation `[N, 8x8]` to represent a `N` chip system,
each with an 8x8 physical grid.

`#tt.device` in is simplest, single chip form `[1, 8x8]`, just maps directly 1-1 to the
underlying physical hardware device.

```mlir
#tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, meshShape = 1, chipIds = [0]>
```

Let's break down what each of these attributes mean:
- `workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>`: This is a 2D logical grid with dim 8x8.
  It's followed by an affine map `(d0, d1) -> (0, d0, d1)` that provides a mapping
  from the logical grid to the physical grid.  In this case, the logical grid is the same
  as the physical grid, so the mapping is the identity function. The logical
  grid can have any rank, but the physical mapping is always 3D, with the first
  being the chip index, followed by the 2D physical core index within the chip.
- `meshShape = 1`: A shape provided as part of the `DeviceAttr` constructor that
  describes the virtual layout of the chips with respect to each other. Note that
  in a multi-chip system, this grid encapsulates the entire system's grid shape,
  e.g. 8x16 grid could be made up of a 1x2 mesh of chips side-by-side. The mesh
  attribute configures how the above grid/map attributes are created such that they
  implement this mesh topology.
- `chipIds = [0]`: This is a list of chip indices.  These chip indices directly reference
  the same chip indices in the system descriptor. The `SystemDesc` attribute
  that this is in reference to is tagged on the top level `ModuleOp`.

Specific examples that this document will cover:
- [Data Parallel Over Batch](#data-parallel-over-batch)
- [Data Parallel Over 2d](#data-parallel-over-2d)
- [Data Parallel Over 2d and Batch](#data-parallel-over-2d-and-batch)
- [Pipeline Parallel](#pipeline-parallel)
- [Reinterpreted Grids (Transpose)](#reinterpreted-grids-transpose)
- [Reinterpreted Grids (Training Usecase)](#reinterpreted-grids-training-usecase)
- [Reinterpreted Grids (Extra)](#reinterpreted-grids-extra)

> Before we move on to more complex examples, it's worth having on hand:
> - The python test `test/python/device_attr.py` which shows how all of these examples
>   can actually be programmed for the device attribute.
> - The [Tensor Layout](./tensor-layout.md) spec as the following examples
>   will demonstrate how tensor layout interacts with the logical device grid.

> **Note on Data Parallel**: There is existing literature that explicitly distinguishes
> between data parallel and tensor parallel, oftentimes describing data parallel
> as duplicating the model across multiple devices and trivially dividing up the batch
> whereas tensor parallel refers to tensor data being distributed and potentially
> communicated between devices during execution.  While this is true for multi-GPU/CPU
> systems, it is somewhat of an implementation detail and given the flexibility
> of tenstorrent hardware there is an opportunity to generalize this concept. In this
> document we will use the term data parallel to refer to any form of parallelism that
> divides any dimension of the tensor across multiple cores/chips.

> **Note on Constraints**: Many of the examples below require careful virtualization
> of the underlying physical system, i.e. some device configurations might
> only work if the chips are connected via ethernet and with a particular
> topology, but these constraints are
> outside the scope of the examples and will be discussed further in the
> [Backend Lowering and Constraints](#backend-lowering-and-constraints) section.

### Data Parallel Over Batch

Given a 2 chip system, `[2, 8x8]`, we can represent a simple data parallel
logical grid that divides the batch dimension in half across the two chips.
This is denoted by `meshShape = 2x1x1` which means the logical grid is 3D.

```mlir
#tt.device<workerGrid = #tt.grid<2x8x8, (d0, d1, d2) -> (d0, d1, d2)>, meshShape = 2x1x1, chipIds = [0, 1]>
```

The affine map here is just identity, so dims `d1` and `d2` directly index
the physical grid and `d0` indexes the chip.

Now we can consider some tensor that, importantly, has a grid of the same rank as
the logical device grid:

```mlir
tensor<16x3x64x128xf32,
  #tt.layout<(d0, d1, d2, d3) -> (d0, d1 * 64 + d2, d3),
    undef,
    <2x2x4>,
    memref<8x3x1x!tt.tile<32 x 32, bfp_bf8>, #tt.memory_space<l1>>
  >
>
```

If we map this tensor onto the above device, it will span across both chips,
half of the batch dimension on each chip.  Within each chip the tensor occupies
a 2x4 grid out of the 8x8 physical grid available.

### Data Parallel Over 2d

In this example we will consider a 2 chip system, `[2, 8x8]`, and view it as
though the two chips are concatenated together side by side to form a single
`8x16` grid. This is denoted by `meshShape = 1x2` which means to concatenate
the chips in the second dimension.

```mlir
#tt.device<workerGrid = #tt.grid<8x16, (d0, d1) -> ((d0 floordiv 8) * 2 + d1 floordiv 8, d0, d1 mod 8)>, meshShape = 1x2, chipIds = [0, 1]>
```

Here we can see that the affine map encodes an indexing pattern such that when
we extend past 8 cores in the second dimension, we wrap around to the next chip.

Now we can consider some tensor that, importantly, has a grid of the same rank as
the logical device grid:

```mlir
tensor<256x1024xf32,
  #tt.layout<(d0, d1) -> (d0, d1),
    undef,
    <4x16>,
    memref<2x2x!tt.tile<32 x 32, bfp_bf8>, #tt.memory_space<l1>>
  >
>
```

This single tensor maps trivially onto the logical grid, spanning the upper
half. Decoupled from the tensor's layout, under the hood the tensor is actually
physically spanning across two chips.

### Data Parallel Over 2d and Batch

The previous 2 examples can be composed together to form a logical grid that
divides tensor across multiple dimensions.  Here we will consider a 4 chip
system `[4, 8x8]` and view it as a `2x8x16` grid. Note that the `meshShape` is
`2x1x2` which means to concatenate the chips in the first and third dimensions.

```mlir
#tt.device<workerGrid = #tt.grid<2x8x16, (d0, d1, d2) -> (d0 * 2 + (d1 floordiv 8) * 2 + d2 floordiv 8, d1, d2 mod 8)>, meshShape = 2x1x2, chipIds = [0, 1, 2, 3]>
```

We can evaluate the affine map to see that the chips are interpreted in chunks of
two, where groups `[0, 1]` and `[2, 3]` each form 8x16 grids and these 2 groups
concatenate to form a 2x8x16 grid.

We can consider the following tensor to map onto this grid:

```mlir
tensor<64x256x1024xf32,
  #tt.layout<(d0, d1) -> (d0, d1),
    undef,
    <2x4x16>,
    memref<32x2x2x!tt.tile<32 x 32, bfp_bf8>, #tt.memory_space<l1>>
  >
>
```

### Pipeline Parallel

Pipeline parallel in the scope of this spec isn't particularly interesting, it
is intended to be used in conjunction with the `ttir.pipeline` operation which
will group sections of the module's operations into groups to form pipeline regions
and will be covered in a separate spec.

What we can demonstrate here is how we can take multiple non-overlapping views
of the system descriptor to form distinct virtual devices.

Given an 8 chip system `[8, 8x8]`, we can form two virtual devices that each
take 4 chips and interpret them differently (though they could take the same
logical grid).

```mlir
#tt.device<workerGrid = #tt.grid<2x8x16, (d0, d1, d2) -> (d0 * 2 + (d1 floordiv 8) * 2 + d2 floordiv 8, d1, d2 mod 8)>, meshShape = 2x1x2, chipIds = [0, 1, 2, 3]>
#tt.device<workerGrid = #tt.grid<16x16, (d0, d1) -> ((d0 floordiv 8) * 2 + d1 floordiv 8, d0 mod 8, d1 mod 8)>, meshShape = 2x2, chipIds = [4, 5, 6, 7]>
```

### Reinterpreted Grids (Transpose)

One particularly interesting usecase that logical grids could enable is to
reinterpret the grid as a form of data layout optimization. For example, if we
wanted to transpose a tensor, instead of having to move the data around to
implement transpose, we could instead reinterpret the grid as being transposed,
leveraging the fact that the relevant data is already located on the correct
cores/chips.

To keep things simple, let's consider a 1 chip system `[1, 8x8]`, but it's not
too big a leap to see how this could map to multi-chip where the cost of moving
data is even higher.


Let's also consider a simple (totally contrived) eltwise unary graph:

```python
a = exp(a)
aT = transpose(a)
relu(aT)
```

1. We'll establish a regular, single chip, identity logical grid:
```mlir
#tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, meshShape = 1, chipIds = [0]>
```
2. Execute `exp`.
3. We'll reinterpret the grid as transposed:
```mlir
#tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d1, d0)>, meshShape = 1, chipIds = [0]>
```
4. _Execute_ `transpose`.  Note that each core only needs to transpose their
   data locally.  Eventually this could be implemented as a no-op by reindexing
   the tile visitation order of the successive operation.
5. Execute `relu`.

It's important to note that we effectively implemented transpose without moving
data anywhere.

### Reinterpreted Grids (Extra)

For the sake of examples, here's a few more ways of reinterpreting the logical grid.

#### Extra Wide Grid
```mlir
#tt.device<workerGrid = #tt.grid<1x64, (d0, d1) -> (0, d0 * 8 + d1 floordiv 8, d1 mod 8)>, meshShape = 1, chipIds = [0]>
```

#### Extra Tall + Transposed Grid
```mlir
#tt.device<workerGrid = #tt.grid<64x1, (d0, d1) -> (0, d1 * 8 + d0 floordiv 8, d0 mod 8)>, meshShape = 1, chipIds = [0]>
```

#### Staircase
```mlir
#tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, (d0 + d1) mod 8)>, meshShape = 1, chipIds = [0]>
```

This could be an interesting starting position for data in implementing matmul as a
systolic array in a ring topology.

## Lowering to TTNN

While the above device attribute encoding is quite flexible, this does not
necessarily mean the target backend can actually support all of these
interpretations.  TTNN backend will be constrained to support only the
specialized grid topologies that are supported by the API.

### Grid/Shard Orientation

TODO

### Multi-device

Please refer to [TTNN Mesh Programming Docs](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md)
for more information on how to program multi-device systems with TTNN API.

Multi-device TTNN dialect will try and stay as close to the TTNN API as
possible.  Let's consider what this looks like from the compiler and runtime
perspectives:

#### Compiler

- **Device Creation**: The TTNN device in the compiler is exactly the same attribute
  from the ttir dialect.  It will encode the `meshShape` into the flatbuffer
  which can be directly used to program `::ttnn::MeshShape`.
- **Tensor Layout**: Again, the tensor layout is inherited in TTNN dialect from the
  ttir dialect.  The grid attribute in the tensor layout can be trivially
  divided by `meshShape` to determine the shape of the tensor slice on each device.
  Broadcasting rules can be applied to determine which [Distribution Strategy](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md#3-distributing-tensor-to-meshdevice)
  to use:
    - **Sharding**: If the tensor grid is > 1 along the `meshShape` dimensions,
      the tensor will be sharded across the mesh devices.
    - **Replication**: If the tensor needs to be broadcasted for this op, by
      extension the tensor layout will be replicated across the mesh devices.

#### Runtime

- **Device Creation**: The ttnn runtime will wholesale switch to working with
  mesh devices via api `ttnn::multi_device::open_mesh_device`, this is possible
  because a 1x1 mesh device is a valid single device. The mesh shape during
  device open will always be `1xN` where `N` is the number of deviceIds in the
  array.  Note that this shape can be reinterpreted by flatbuffer programs on
  the fly with `SubMesh` API.
- **Tensor Creation**: Tensor creation in a multi-device system is a bit more
  involved.  In order to upload a multi-device tensor to the mesh, the host
  tensor much first be created with `MultiDeviceHostStorage`.  The ttnn runtime
  can automatically do this during `handleToHostMemoryConfigOp`:
    - Regular host tensor will bounce through new tensor with
      `MultiDeviceHostStorage` type.
    - `tensor.to(mesh_device)` will allocate/move the tensor to the mesh device.

## Lowering to TTMetal

In TTMetal dialect we are only constrained by what we've implemented in the
tt-mlir compiler, this means it is much more flexible and can theoretically
support any of the grid interpretations above.

## Test Plan

- `test/python/device_attr.py` covers all of the examples above and asserts the
  IR is correctly generated.
- Additional functional unit tests will be added as op and runtime support is
  added.

## Concerns

- `tt.device` is very flexible, but with this flexibility comes the potential
  for misuse.  It's important that the compiler is able to validate the legal
  configurations of this attribute for the target backend.
