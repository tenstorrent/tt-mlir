# D2M Dialect v2.0

# Goals

Provide the ability to abstractly represent high level computation on tenstorrent devices which is lowered through tt-metal APIs.

Provide a standard entry point for multiple dialects and flows within the MLIR ecosystem.  
tt-lang, triton (through triton-to-linalg), polymage, standard dialects: linalg, affine, etc

# Key Concepts

Entry point for d2m flow starts with Tensors & Layouts (currently with views) as inputs to d2m.generic op in either *Implicit Blocked Form* or *Unified Explicit Form*.

In *Implicit Blocked Form*, the front end optimizer is able to choose blocking and Tensor Layouts to optimize the usage for SRAM buffers.

In *Unified Explicit Form … TODO*

Data movement is represented through the use of explicit d2m ops (remote\_{load, store}, TBD local\_allgather, remote\_dev\_{load,store}). Data movement / Compute pipelining is achieved through the splitting of these ops to data movement and compute threads. Resource allocation (Threads, Semaphores, CBs) is also determined algorithmically during lowering, but pinning from the top-level ops is also possible.

# ![][image1]

TODO: Discuss

- Single thread top-level IR & Thread splitting / division of work

- Gaps ?:  
- Multidevice representation  
- Multihost representation

# 

# Valid IR Forms

These forms are roughly laid out progressing from **higher level forms** to **lower level forms**, following the natural conversion order of the full D2M compiler pipeline for a typical `ttir` operation. The high level characteristics are gradually lowered to get to the fully lowered generic form.

**High Level Characteristics**:

- Simpler high level region ops  
- Unified single-thread only  
- Can be in tensor form as well as memref  
- Implicit blocking loops derived from operand shapes and generic op attributes  
- Operand grid shapes are constrained by indexing maps

**Low Level Characteristics:**

- Explicit blocking loops with concrete bounds  
- Only generic op attributes are grid and thread types  
- Memref-only  
- Internal buffers are represented and allocated  
- Multiple compute/dma threads

# 

## 

## Implicit Blocked Form (High-Level)

| *Frontend pipeline, Grid Selection, Bufferization Input* |
| :---- |

- Only form with indexing maps and iterator types  
  - Permits safe tensor layout selection  
- Only form with Tensor operands  
- Define Tensor lifetimes, locations  
- Define scratch buffer sizing based on d2m.generic definition including fusions  
- d2m.generic with optional “lowering plan”  
  - Blocking sizes  
  - Loop interchanges  
  - Intermediate sizing

```c

    %7 = d2m.generic {
       block_factors = [1, 1], grid = #ttcore.grid<1x1>, 
       indexing_maps = [...], iterator_types = [#itParallel, #itParallel], threads = [#d2m.thread<unified>]
    }
    ins(%arg0, %arg1 : !device_tensor, !device_tensor) 
    outs(%6 : !device_tensor)  
    {
    ^unified0(%cb0: !block_cb, %cb1: !block_cb, %cb2: !block_cb):
      // Indexing of outer loop is abstracted into d2m.block_index ops; this
      // corresponds 1:1 with indexing maps for each load/store op's operands
      %block0 = d2m.block_index(0) : index
      %block1 = d2m.block_index(1) : index

      %10 = tensor.empty() : !block_tensor
      %11 = d2m.remote_load %10 %arg0[%block0, %block1] : !block_tensor, !device_tensor -> !block_tensor
      %12 = tensor.empty() : !block_tensor
      %13 = d2m.remote_load %12 %arg1[%block0, %block1] : !block_tensor, !device_tensor -> !block_tensor

      // per-worker compute loops are often defined as linalg.generic ops
      %14 = tensor.empty() : !block_tensor
      %15 = linalg.generic {...} ins(%11, %13 : !block_tensor, !block_tensor) outs(%14 : !block_tensor) {
      ^bb0(%in: !tile, %in_4: !tile, %out: !tile):
        %17 = "d2m.tile_add"(%in, %in_4) : (!tile, !tile) -> !tile
        linalg.yield %17 : !tile
      } -> !block_tensor

      %16 = d2m.remote_store %6[%block0, %block1] %15 : !device_tensor, !block_tensor -> !device_tensor
      d2m.yield %16 : (!device_tensor)
    } : !device_tensor
```

## 

## Affine Blocked Form

| *Loop Fusion, Scalar Replacement, Allocator Input* |
| :---- |

- Memref-only  
- Outer blocking loops are present, but loop bounds must be `BlockFactor(constant)`  
  - i.e. `affine.for %arg0 = 0 to BlockFactor(0) …`  
- Block factors, indexing maps, and iterator types are still present.  
- Loop interchange is fixed and interchange attribute removed.  
- **Generic Op fusion is done in this form.**  
- Allocator would determine blocking factors for generic ops in this form

```c
   d2m.generic {block_factors = [1, 1], grid = #grid1x1, indexing_maps = [...], iterator_types = [#itParallel, #itParallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg1 : !deviceMemref, !deviceMemref) outs(%alloc : !deviceMemref)  
   {
    ^unified0(%cb0: !cbTileMemrefL1, %cb1: !cbTileMemrefL1, %cb2: !cbTileMemrefL1):
      %block_factor0 = d2m.get_block_factor(0) : index
      %block_factor1 = d2m.get_block_factor(1) : index
      affine.for %arg2 = 0 to %block_factor0 {
        affine.for %arg3 = 0 to %block_factor1 {
          %block_offset0 = d2m.block_offset(0) : index
          %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg2)[%block_offset0]
          %block_offset1 = d2m.block_offset(1) : index
          %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg3)[%block_offset1]

          %alloc_0 = memref.alloc() {alignment = 64 : i64} : !tileMemref
          %2 = d2m.remote_load %alloc_0 %arg0[%0, %1] : !tileMemref, !deviceMemref -> !tileMemrefL1
          %alloc_1 = memref.alloc() {alignment = 64 : i64} : !tileMemref
          %3 = d2m.remote_load %alloc_1 %arg1[%0, %1] : !tileMemref, !deviceMemref -> !tileMemrefL1

          %alloc_2 = memref.alloc() {alignment = 64 : i64} : !tileMemref
          linalg.generic {...} ins(%alloc_0, %alloc_1 : !tileMemref, !tileMemref) outs(%alloc_2 : !tileMemref) {
          ^bb0(%in: !tile32x32f32, %in_3: !tile32x32f32, %out: !tile32x32f32):
            %5 = "d2m.tile_add"(%in, %in_3) : (!tile32x32f32, !tile32x32f32) -> !tile32x32f32
            linalg.yield %5 : !tile32x32f32
          }

          %4 = d2m.remote_store %alloc[%0, %1] %alloc_2 : !deviceMemref, !tileMemref -> !deviceMemref
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
```

## 

## Unified Explicit Form

- Memref-only  
- Outer `scf.for` blocking loops are present; blocking factors are fixed.  
- Post Allocator/Optimizer, all generic ops are normalized to unified explicit form.  
- Block factors from Explicit Loop Form are substituted into region body and removed from attribute list  
- High level DMA constructs must be lowered e.g. high-level multicast load  
- Relevant passes  
  - Internal `linalg.generic` incrementally lowered to affine  
  - Affine loop optimization  
  - Affine loop manipulation for fusion and dest reg constraints  
- **Tt-lang, kernelize entrypoint?**

```c
   d2m.generic {block_factors = [], grid = #grid1x1, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg1 : !deviceMemref, !deviceMemref)
        outs(%alloc : !deviceMemref)  {
    ^unified0(%cb0: !cbTileMemrefL1, %cb1: !cbTileMemrefL1, %cb2: !cbTileMemrefL1):
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
          %0 = arith.addi %arg2, %core0 : index
          %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
          %1 = arith.addi %arg3, %core1 : index

          %alloc_0 = memref.alloc() {alignment = 64 : i64} : !tileMemrefL1
          %2 = d2m.remote_load %alloc_0 %arg0[%0, %1] : !tileMemrefL1, !deviceMemref -> !tileMemrefL1
          %alloc_1 = memref.alloc() {alignment = 64 : i64} : !tileMemrefL1
          %3 = d2m.remote_load %alloc_1 %arg1[%0, %1] : !tileMemrefL1, !deviceMemref -> !tileMemrefL1

          %alloc_2 = memref.alloc() {alignment = 64 : i64} : !tileMemrefL1
          linalg.generic {...} ins(%alloc_0, %alloc_1 : !tileMemrefL1, !tileMemrefL1) outs(%alloc_2 : !tileMemrefL1) {
          ^bb0(%in: !tile32x32f32, %in_3: !tile32x32f32, %out: !tile32x32f32):
            %5 = "d2m.tile_add"(%in, %in_3) : (!tile32x32f32, !tile32x32f32) -> !tile32x32f32
            linalg.yield %5 : !tile32x32f32
          }

          %4 = d2m.remote_store %alloc[%0, %1] %alloc_2 : !deviceMemref, !tileMemrefL1 -> !tileMemrefL1
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
```

## 

## Memory Feasible Form (Fully Allocated Form?)

| *Allocator Output, Affine Loop Manip, DST Optimization, Scratchpad Use, DMA-lowering Input* |
| :---- |

- This is just like Fully Blocked Loop Form, but all `memref.alloc` are assigned addresses  
- Harden allocation and memory addresses  
- This should never fail if planning was correct  
- If there was no plan, or we entered the pipeline after planning, we could fail memory allocation

## 

## Split Explicit Form

| *Mostly internal to DMA lowering passes* |
| :---- |

- Memref-only  
- Remote loads and stores are in *explicit implementation form* → assigned a specific CB, scratchpad buffer  
- Middle-end pipeline lowers loads and store ops to optimized low-level DMA ops in distinct DMA thread regions  
  - Each Compute and DMA thread region bound to subset of Tensix baby-RISC cores  
  - Load/store ops scheduled (partitioned) onto N independent DMA threads, one per datamovement RISC  
  - DMA thread regions produce/consume compute thread’s IO via CB/scratchpad buffers.  
  - Non-contiguous load/store ops are optimized to coalesced gather/scatter loops   
  - Double buffering optimization (maximize DMA-compute overlap)

```c
d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
    ins(%stream, %stream_2 : !streamMemRef, !streamMemRef)
    outs(%alloc : !outMemRef)  {
^datamovement0(%cb0: !cbType, %cb1: !cbType, %cb2: !cbType):
  %core0 = d2m.core_index(0) : !indexType
  %core1 = d2m.core_index(1) : !indexType
  // this remote_load will be lowered to low-level DMA gather loops in this form
  d2m.remote_load %stream[%core0, %core1] into %cb0 : !streamMemRef into !cbType
}, {
// Each datamovement thread is bound to a physical baby RISC core
^datamovement1(%cb0: !cbType, %cb1: !cbType, %cb2: !cbType):
  %core0 = d2m.core_index(0) : !indexType
  %core1 = d2m.core_index(1) : !indexType
  d2m.remote_load %stream_2[%core0, %core1] into %cb1 : !streamMemRef into !cbType
}, {
^compute0(%cb0: !cbType, %cb1: !cbType, %cb2: !cbType):

  scf.for %arg2 = %c0 to %c1 step %c1 {
    scf.for %arg3 = %c0 to %c1 step %c1 {
      // loads and stores become wait/reserve from CBs in compute thread
      %0 = d2m.wait %cb0 : <!cbMemRef> -> !cbMemRef
      %1 = d2m.wait %cb1 : <!cbMemRef> -> !cbMemRef
      %2 = d2m.reserve %cb2 : <!cbMemRef> -> !cbMemRef

      // ... per-worker compute loops go here ...

      // correct CB push-pop inferred at thread split time
      d2m.push %cb2 : <!cbMemRef>
      %3 = d2m.wait %cb2 : <!cbMemRef> -> !cbMemRef
      d2m.pop %cb2 : <!cbMemRef>
      d2m.pop %cb0 : <!cbMemRef>
      d2m.pop %cb1 : <!cbMemRef>
    }
  }
}
```

## External Symbol Form

| *Backend form* |
| :---- |

In this form, the generic op has no regions. Each thread attribute refers to an external `func` symbol that represents a compute or datamovement kernel.

* If using metal backend, lower to:  
  * d2m.dispatch  
  * TTKernel split into threads on dispatch op  
* If using ttnn backend, lower to:  
  * ttnn.generic:  
    * Cb programing  
    * Kernel  
  * ttnn.empty

```c
func.func private @datamovement_kernel0(%arg0: !l1_cb, %arg1: !l1_cb, %arg2: !l1_cb) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  %c0 = arith.constant 0 : !idx
  %core0 = d2m.core_index(0)  : !idx
  %core1 = d2m.core_index(1)  : !idx
  %0 = d2m.reserve %arg0 : <!l1_memref> -> !l1_memref
  // ... coalesced gather loop goes here
  d2m.dma_wait %tx
  d2m.push %arg0 : <!l1_memref>
  return
}
func.func private @datamovement_kernel1(%arg0: !l1_cb, %arg1: !l1_cb, %arg2: !l1_cb) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  %c0 = arith.constant 0 : !idx
  %core0 = d2m.core_index(0) : !idx
  %core1 = d2m.core_index(1) : !idx
  %0 = d2m.reserve %arg1 : <!l1_memref> -> !l1_memref
  // ... coalesced gather loop goes here
  d2m.dma_wait %tx
  d2m.push %arg1 : <!l1_memref>
  return
}
func.func private @compute_kernel2(%arg0: !l1_cb, %arg1: !l1_cb, %arg2: !l1_cb) attributes {d2m.thread = #d2m.thread<compute>, tt.function_type = "kernel"} {
  %0 = d2m.wait %arg0 : <!l1_memref> -> !l1_memref
  %1 = d2m.wait %arg1 : <!l1_memref> -> !l1_memref
  %2 = d2m.reserve %arg2 : <!l1_memref> -> !l1_memref

  // ... compute loops go here ...

  d2m.push %arg2 : <!l1_memref>
  %7 = d2m.wait %arg2 : <!l1_memref> -> !l1_memref
  d2m.pop %arg2 : <!l1_memref>
  d2m.pop %arg0 : <!l1_memref>
  d2m.pop %arg1 : <!l1_memref>
  return
}
```

# 

# Examples

- Matmul \+ Bias  
- Convolution Lowering from stable hlo  
- SDPA  
- RMS Norm / layer norm/ group norm

# Ops

Core and Layout

* `d2m.view_layout`   
* `d2m.to_layout`   
* `d2m.empty`   
* `d2m.full`   
* `d2m.mesh_shard` 

Dispatch

`d2m.generic`

D2m.generic

- What advantages do we gain from using d2m.generic over linalg.generic when cbs are pushed lower into the stack?  
-   
- **d2m.generic:** Explicitly models execution over a **Grid of Cores**. It introduces the concept of "SPMD" (Single Program, Multiple Data) at the IR level  
- **d2m.generic:** often carries `block_factors` attributes that dictate how tensors are broken down into micro-blocks to fit into the L1 scratchpad memory.

Tile ops

### **Data Movement and DMA**

* `d2m.dma_write` D2MGenericRegionOps.td:909-973  
* `d2m.dma_read` D2MGenericRegionOps.td:975-1020  
* `d2m.remote_load` D2MGenericRegionOps.td:1058-1150  
* `d2m.remote_store` D2MGenericRegionOps.td:1290-1410  
* `d2m.store` D2MGenericRegionOps.td:1791-1858

### **Circular Buffer**

* `d2m.reserve` D2MGenericRegionOps.td:1687-1709  
* `d2m.wait` D2MGenericRegionOps.td:1663-1685  
* `d2m.push` D2MGenericRegionOps.td:1711-1789  
* `d2m.pop` D2MGenericRegionOps.td:1711-1789

### **Indexing**

* `d2m.iter_index` D2MGenericRegionOps.td:1885-1890  
* `d2m.block_index` D2MGenericRegionOps.td:1892-1901  
* `d2m.core_index` D2MGenericRegionOps.td:1903-1938

### **Remote Access**

* `d2m.get_global_operand` D2MGenericRegionOps.td:1944-1990

### **Write Mask Tile Ops**

* `d2m.write_row_mask_tile` D2MGenericRegionOps.td:1996-2011  
* `d2m.write_col_mask_tile` D2MGenericRegionOps.td:2013-2028

### **Destination Control**

* `d2m.acquire_dst` D2MGenericRegionOps.td:876-893

### **Data Movement and DMA**

* `d2m.dma_write` D2MGenericRegionOps.td:909-973  
* `d2m.dma_read` D2MGenericRegionOps.td:975-1020  
* `d2m.remote_load` D2MGenericRegionOps.td:1058-1150  
* `d2m.remote_store` D2MGenericRegionOps.td:1290-1410  
* `d2m.store` D2MGenericRegionOps.td:1791-1858

### **Circular Buffer**

* `d2m.reserve` D2MGenericRegionOps.td:1687-1709  
* `d2m.wait` D2MGenericRegionOps.td:1663-1685  
* `d2m.push` D2MGenericRegionOps.td:1711-1789  
* `d2m.pop` D2MGenericRegionOps.td:1711-1789

### **Indexing**

* `d2m.iter_index` D2MGenericRegionOps.td:1885-1890  
* `d2m.block_index` D2MGenericRegionOps.td:1892-1901  
* `d2m.core_index` D2MGenericRegionOps.td:1903-1938

### **Remote Access**

* `d2m.get_global_operand` D2MGenericRegionOps.td:1944-1990

### **Write Mask Tile Ops**

* `d2m.write_row_mask_tile` D2MGenericRegionOps.td:1996-2011  
* `d2m.write_col_mask_tile` D2MGenericRegionOps.td:2013-2028

D2m.acquire\_dst

- Scoped  
- 

d2m.iter\_index  
D2m.core\_index  
D2m.dma\_read  
D2m.dma\_wait  
D2m.dma\_write  
D2m.semaphore\_{set,inc,wait}

New  
d2m.dispatch \<- only allowed tensor as IO?  
d2m.remote\_load  
D2m.remote\_store

- Implicit synchronization  
- 

d2m.scratch\_allocate

## Tile Ops

d2m.tile\_add  
d2m.block\_matmul  
…

d2m.wait  
d2m.reserve  
d2m.stream\_layout

# 

# Unanswered questions

- Handling of tiled / non tiled data  
- Spatial op  
- Subdevice  
- 

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAACTCAYAAAAOXCSmAAA3zUlEQVR4Xu2dh5sUxd6237/jfb9LzzEdOGYBQUCCIAhIBgkKBiSIBImSQTIGxMABAQVBPAQDIIjknJEgSA6b8y5hWWBjffNUb/V2V8/s9sxsmJ597uv6XV1dVd09Pb0zc291ddX/CEIIIYQQ4in+R88ghBBCCCGRDQWOEEIIIcRjUOAIIYQQQjwGBY4QQgghxGNQ4AghhBBCPAYFjhBCCCHEY1DgCCGEEEI8RqUI3Fu/DxMt176uZ1cqLdb01LMqlNofHRFxmff0bPH/hu/Vs8Snf8TI5dfb48W4ny5rpeGRV1Ak/neY85iEEEKim8Nbb+lZrrl06q6eFRJbV2XqWZINSzPElTN3xbFdt21x3BdX/3bmnzueI7dLuHbf3EdmSp64lVVgrpcHv//g//Uqsm+UHC/Dd/xAHNt5W88qdypc4J5Z9rLIyQvvD+Gppc31rKDIK8wPex/lhT+ZmrPpurH8PUaMXHXJXhgm9/ML/R6TEEJIdNO/6UU9yzXhbGvF335+WZQu5g6PN9cLC4r81tu9/qaYN7Kk3sm92eKTIXEyPfnN6zLKG3+vw8qWVVlymZdbJGIuOBtpQGFhkRjSKrjfcv24g18pe/sKFThIk4pRu6bL5VubhsplQVGBeHppC5muv6KdrH/tZqxoteYNc5vs3DvijY2DzfX2P78t0u9mmuvfn1kjt3t2WUux/OxPMq/Jj10cx1/qq1feAjdw+XlRf+Yx8e/xB6UgPTbmgHjEF/83Yp8sn77hmsi+VyAKCo0WsCcnHhLPTjlsylTtqUfE/w3fKx7+cL9o+vFxmWcVuNiMe7Lu87562BZkZOeJB0fuk+v/GntA5qEO9vP4hEPmvtUxkYe6FDhCCKl+QAriLt8X0/vGyNYuxZnDd8TM/jEiNT7XzENr0uyBsSLmoiElSihy7xeZrVIQFmx3/k+jNSw5NlckXrsvvhgdLw7+UdLalxSTK2YNiJXH1MUEIA/SpghW4EZ1virG97hmqSnE9p+yxOcj4sXNjHy5vmlFptzv5N7Xxf17ReLA7zfF3t9u2lrY7uUUim8mJ4h136aLouKX4+91gP9MSBRr56eZAldYKMTfx4z3IfF6rpg7LE5sX6vK7AIXf+W+mNE/Vpw+kG3mpSUa73fC9fti6+osedx1S9LNFkWsq9cUiAoVOGAVJ6QPJ56Q6VrfvyJ+ufi7TL+/ZZy4nHVdnMu4ZNbfcm23mdb3AfkDdZa3NvNmHfrKTOcX5osVf/8sNl/bKfM+PbKwQgSuSbF4PeCTtsV7EmVaydLU9dfEbZ/Avbv0nHhtwV8y77dT6Wa5VaogWsAqcChH65lKKylTtP78pFkGsVNp0P7LU2LIjxdkes+FLAocIYRUQyABi6cmyVuNSOO23q3MfLFtTZa47RMFJSsQEKSz0vLNljGsZ9806txIzxdrfPIytM1lKT2fDDZawRZ9lCTLsf8PXr3sE5AMc9sUnxwqMdHR84IRuEEtL4kvP0yw1PL9Hr98SWzzHQuvd8BLxn6wP0R6cp74+2iOTF84mSPmjUoQE3tdN+vg3FLicsV7zUq20xnZ6Yq4cCJHnpMqv3T6rpj1XqxMfzczWYrjxuUZUrqsAodbxcPbXRH37xaKBRMTRaGvQmZqvvl+zx4YJ5fytSbliYJiscX6nVul3x6udIHzlwbzTyyzCVzKnbSAAtf4x84y/JXXXfGqSMvJsOXhFq5+vHCBwM3+/bpMP+ETsDu5xhutCxxa6NKLBQuXRZU/VtyCBib9elUudYGrOe6gGem38xx5qbdybXKGlji17YVk4z+D7PsFFDhCCKmGWGXk7NE75m05tJAd3HxTDPBJS4ZPcIa3v+K4HagECLcK1fqhrbdk7N90U1w+c1cK3OJpSbIccgj5gxjuXnfDth8dPS8YgZv6ToyjrvW1fTcjWRQVGnlKhiBww9oZ/cshRUNaG+lp78aY26l96vvW81Z8miKXVoGDvKG/4dZVWeLU/mybwEEMD24xjoHY5JM87A+toFb042JdtSYGosoE7ullL4urN4yTX3Die3Es+ZRrgVMUFBrS5E/gZhz8Qi7B6vPrq0zgVh5Kli10oPHs47YWuKSb98Wtu/lm3p6LWWZrHGTs6x3GH+6vf6b5rF2IBjOOiXNJxkXfed5oqvUncBtOpcl8/Cfw7JQjFDhCCKmGWKUALVRoQRvZ8YpPjAzBQmsVBA59yTZ8Z/xeKrDt6C5XxYgOV8z1gnz7PT0I3MrPU2Ua4gKBy7ldIH6Ya0gO0MXEX14wAodbqPhtQ/1dvxrn4W9bax4EDucN0IKoBA63MHXK2tdH7xgPHSqBw3sy4XXjdi5ui57cl+17fUXi/RaGwKFlEu+xFRxf3YpV6MfFOloUS6PKBO5OXo5cVwEuZF4x09ZWtIUnl5v19sQfkstnlhn954B1v/VWtBWZ927I1i61Tedf3y13gUNr2aErN2W61dyT5r3qB0YasnbeJ1oqb+KvV6VEXUzOkf3kwPX0uzIP8YNP8hQ1xx8UQ380LuSr807K8q93GM3VAPtH3qAfjFuk1qda/zlqv5me8MsVWe9yag4FjhBCqiGQANxeHNv9milgG5enS7nAbUjkQS7y8wyBgsjJW5DFggTea35Jyg/6xqk6kLqcOwV+BU4dd3zPa2JIq8sOMVHlaCVTBCtwCmyz+qs02T8N54m+fkqc3Agc6qj+dKo+lhBXK5tXZoph7a7IuqqetQUOebhVjSUETuWN7XZV3Mk2bkPj/cCt2CPbb8kWOuRN6lVy3IWTE+VxZ/QzBBH5qgUxEBUucIQQQgipfNDvCqBlSOfO7QLZEd+KtZ61te3unZKK6KcVqHM9xESh9qVegxWIlvWhB3A321kP6MfSX7N6nahnff14+MKKdT+590t2kuMTLOvQICA1wfl+3cosMG8nK6z7RF86HdyqVqAvXvFNw5I87bpgH0ps/QmtDgWOEEIIIZUG+qG5EZTqzPpv0/UsBxQ4QgghhBCPUWECd/P+LVsft2gIMG9rnNl3jcFgMBiMSAqA1i0vh3W2BRKYChW4aKLO8jZyCYHDk6OEEEJIJGEVOK9yZNstCpxLKHAuocARQgiJZChw1QsKnEsocIQQQiIZClz1ggLnEgocIYSQSIYCV72gwLmEAkcIISSSocBVLyhwLqHAEUIIiWQocNULCpxLKHCEEEIiGQpc9YIC5xIKHCGEkEiGAle9oMC5hAJHCCEkkqHAVS9cCdyq8+vF2D2zzPWBW8eKi1nXZPpw0gkxaNt4s0xBgSOEEEIqDwpc9cKVwKmppO7m3xO/Xtpsm1pKpc+mX7BtQ4EjhBBCKg8KXPXClcCdz7gsNl3dYa6vubBBZOfekenUnHS5rkOBIySyeeCBB6pFRBv6+UVrRBunTp1ynGM0Rs2aNfVTD4rSBO7o9lti8lvXzfUZ/WLEys9TZTo5NldM7HVdFBYUmeXRjiuBCwUKHCGRDb5sn376adG4cZOojBo1aspzjDZwTv/4xz8c5xtNEY3XbfHixfK89HONplASFw6lCdyglpfM1sWiIqOlUa3PHBAr05tWZFo3iWoocC6hwJFoA1+0s+d8IlIzbkRltGvXPuwfk0gE5/T44084zjdaYs/+Q1F53ZTA6ecbTVHRAkfsUOBcQoEj0QYFzptQ4LwJBc4dFDj3UOBcQoEj0QYFzptQ4LwJBc4dFDj3UOBcQoEj0QYFzptQ4LwJBc4dFDj3UOBcQoEj0QYFzptQ4LwJBc4dFDj3UOBcQoEj0YZbgYtPSnXkIRKS0xx5kRQUuLLjaky8OH/xiiO/tDgXZP1ggwIX+RHob4ACV7lEhMBlZmaKjh07RnQ82rCGXNZ7qbVo176Do7ysmDBhgn7aUcHQoUMd5xptsWvXLv20owI3AvdOn3f9/uikpGf5zfcXqAdRsK4jnq9bVzz88MNizsefOrYpK2bOniM6d+nqyLcGBS5w9O3bz7wOKl5//Q1HPX+BukmpGTL9xVfzbX8HV67HOuoHG9VR4PRroUKvp0dMfJJITst05Fd0BHpt6nWHAwXOPREhcI8//rjjDzcaIxrRzzFaIxrBeYUqcOrLWs/zF0eOnTDTGzZuFg8++KC5fv7SVVMGggkKXOgC99umzXIfa3/+1czbun2n6+uJeuqaQeR37ztgK9t/6Ihjm2CiugqcnucmHnzwH2LM2HGO/IqOQK+3PL4vKXDuiRiBe6NXb8cfQ7QEvuzC/aOOVHBeK39c7TjnaImHHnooqq9deQnc2+/0EXGJKaJvvwHi9Td62epNmDRZLqfPmCVq16ktunTtKkZ/ONYMawvC1Zg4uX2fPn0dt26/WbREdOvWXYoHBS50gevzbt+A1xSxbsNGsXjJd+KCT667dH1NjBw12laOba3SPeC9gfJ2Oq4lyvr1HyBGjxnr2K/boMDZo3fvt0RiSrq5Pnb8BLFn7wH5fuOfoeYvvyzTqhzX5q133hG9er8pzvx90cwf+sEw+VkbMXK02do6dtx4cflajPybwG+w9bOI/QwfMUp06txZDB76ge01BXq9yA/32lHg3EOBq4SgwHk3KHDuBE59cTd96SXx9DPPOMrwA9SxYyfxr3/9S9SqVVs0bfqSDMwosKe4BWfXnv2y7oxZs0VXnzhY91G7dh25jh+lGjVqyDQFznnObmLz1u1yHx06dhSXrl53lNes+W9ZjmvToVNnKQn69VQCd+yEMT1UTEKSvJ5I12/QQKb1/bqN6ipwuBbWUO/xP/75T3O7VWt+kumjf5403+8nn3zSfL8vX4uVeSN80v3B8BGO66bqQ7qvxyWaec2bvyz+/W/julvr161bTwzxydujjz5m+40u7TzCvXYUOPdQ4CohKHDeDQqce4FLtrTKYB0tciqtWhA6dOoovv1umVkPPyZK4FBv3pdfmWXduncXu/buF7EJybJs34FDtv1T4Jzn7DbQyqZ+bBHoj5icZlw/JXCqrt7fEWld4Kxl+w8edhwvmKiuAqfHi40ayTK0imEd3Q+w/HL+f8ztINfW1s6HH35EvPJKK3P9hx9XmS3g2BZ9TlWZEjhr616g1+fvb0Cvo/LDvXYUOPdQ4CohKHDejeoucO8PGuz3y/rYidOlfqFjXT24YP2RCCRw6jOix7QZM8Who3869j912gwKnJ9zDjVGfTjGfI91gUNgXd1eQ5oCFzxlCZyepwfq9Oj5ui1PFzjrZ0dFw4YvmmXTfZ8nVVcJnH4MlW7WvLljX/7q6dsjwoEC5x4KXCVEqAI3cuRIPavc6Nevn54VEjgvCpydYOsHQ/369fWskMHrLEvgVId31Zqm4t2+/WTfG7Wuf6FjPRiBU/WOHj9p2w8CQxbo+0cfHgqc85zDCfUeBxI4a7o0gduz/6Bj38EEBc4ZTzzxpLydiXros6by8RDDsOEjbPuZO+8Lx/aqLBiBC6bMmh/utaPAuYcCVwkRrsCdPHlStG3bVuTm5Yr58+eb5UOGDJHLMWPGyOV7770nvv76a5meMmWKDLBw4UKxYMECYyMfR48elX2Rxo4dK9e7du0qtm3bZpYHA86LAmdH1b9//75o3bq1OHHihEhJSRE5OTky/80335TL/fv3y+W3334rBg4cKNNZWVli3759YseOHeLs2bNi0KBBMl+BfY8ePVqmx40bJyZNmmQrDwbsqyyBQ6Ae+kPhVibW8QOBvL/+vmCro28TrMDhvUaoMjy9io70ah8NGjSUaXVLqbwFrjzlWGfFihV6VsjgnMIROLzHTz71lG0cvzp1jD6GSCuBm/eFcTv7y6/tQ4UgPfdzQxL8CVyjFxs7jhlMhCJw06ZN07PKDfV5DZdQBa6z7/tZlT/zzLO2urglis+mWu/QsZOtHJL22+9/mMcIVeBatWodsEzfPthrp0OBc49nBO4xn3As/GaxIz+cOHv+ku0PcfG3S0Wd55931LNGs2bNxaAhQx35pUW4Avfuu+/KJfYB+Tp16pRcqn02aNBASkJhYaFITEyUPxYvvPCCKCgoEJcuXRJ79+6VMmGlZcuWctm+fXu5/Oqrr2TdYMFrCEfgAn0RIJat+EGWL1rynVzfd+CweOyxx0S37j3kWHzv+N4XfZvyjnAEDp3tweDBg0Vubq5o1KiRrfyRRx6R+VeuXJHrL774oti4caOIj4+31UMdhco7cOCAyM7OFsnJySH/wGBfbgQOIvbP4o7UKrp27Waro19HrMf4fiBUWt1+w9NxP/28zqz3zDPPiENHj5vrTz/9tO046knUPfsO2vKnTpsuer/5lu2YeoQjcD/99JPYvXuXTJ8+fVour169apZDyG/cuCH/OcLnDty7d0+sXLlSruPHGp8/xauvvirWrVsnP4fHjx8X33//vVkWLDincATu0tUYs8O6NdRwL0rg1N8+YtToD83tlSQg/ta+Q5ev/NEs+2bxt45ju4lQBG7q1KlmesmSJfK9R9y6ZfwO4XMFUlNT5RL/FP/www8y/csvv4iEhAT5Hbp7927x3//+19iRj8uXL8vXgmsHNm/e7Kv/s1keDGUJnL9QZTt37zPr4rq079BRpq/5PpuqrurfpmRLhfpHC+mv/7PA3I/qW6q/DpXGd63axxu9etnKXmpm3F7Fd5y+PSIcKHDu8YzA4Y9iwICBjvxwYvuuPbY/ymuxCT5ZKP1LZ+Wq1eLUX3/LdFJKuuMD4C9CFTj88AN8iQD1A6OeCsMXVP/+/eUP/MyZM83tAAROgR+UZ5991lIqROPGjeWyYcOGcgnxw5dTsOB1hCpwTzzxhNy+62uvOco6d+lie2+XLTdkTq1v3LxF7D8Y3nhTbiIcgYOQgQ0bNshlvXr1zJa2U6dPSdFWPygK9UOj+OCDD8Tq1avNdbXvRYsWOfKCBdu5ETivRqgCp7a5e/eu/OyghQOg1bqoqEgO7gyWLVsmP4NNmjSR6+ofIH/H7Nu3r5mGLAB8jkMB+w9H4MoKf7dQKzPCETi1HX5T1Po333wj/zFQ67hLkZ6eLq8d/omCzCrZ9vcdqPZZq1YtM++5554z024pTeCiJXB+wV47HQqceyhwYXygcFvIzfahChy2ee01YzgF/HeF/+IBbpXqX1gAsoF6q1atMgUOrW/I038s1D4hf1iG8voAtgtF4NR7EhPvbMZH1KhR03e+bc11jB3mr15FRygCh/ce72n37t3lD4TaHjIACQDWfSKNepDqTZs2yTz8oKDVC5GWlmbWHT58uNx3RkaGLINc3L592ywPBhyXAleCLnAA7z0EHLe0MSNHzZo1zfIzZ87Iz49qIUWLHIDkoVtD586dzf1YBS4vL08u8bcVCjg+Bc6O/n2o/kFF1xN890HY9uzdI7skYPYYfGbUtYPAKdDNAS3lS5cuNfPUPp966ilHXjBQ4NxBgXOPJwUOP4II9ceCuHD5qm3dup36IlexYuV/ZZkucGvW/mxbb9nyFfsxLhnH6PNuPzHQ90VgLbNup0eoAucFcF6hCFzP11833zMsv55f0rSvv6+B4q233jbrT/5oqq1s7/6SISdatGxp5j/kkx88Eq+/nkARisB5BZwXBa4E9C3Ewz24tdatWzd5e1eh9oPlkSNHzDT6oUK+gRI4iPWAAQPMFnSA/o6QCrTw4LsL/5ihdTUUcNyKFDiEekihKiIUgUN9CBL+ycU1bNasmczHLdQ5c+aYdSDXuI2NNK4PbhMqgcM/WBBttLDiH18FurL06tVL3krHNUSL6+HDh81yt1Dg3EGBc48nBU79kZw4dca2bq3buvWrtrIr1+PkOvrNqLq6wKHPhlofNHiITKO/CNYhfeiPgzzV/6CiW+C8AM4rFIHDduoWqJIvvRwjgKt1PPGI/6St5W3a2K+xKnvk0UfNzvB4WtFaho7vZfVztAYFzrsRrMB5BZxTRQtcVUYoAucFKHDuoMC5x9MCp8rq1q0r/1tW6888+6zs86XqYjwbfV9YliZwWE6YaAyAqG9LgSsB5xWswB0/eVq2UqBjuwrsZ/rMWWYdrAcjcJjKSZUN+eAD23V8+eUWslyFm2umggLn3aDAeTMocN4NnF+4144C556oEbgXXqhvrusC17SpfVoXtS0FLnxwXsEKHMY0UtdQD1UH6WAEzjoWki5wH44ZJzb+vsUMTCWkv6ZAQYHzblDgvBkUOO+G+h4PBwqce1wJ3FNLm4uXVnWT6aVnVsn127nZcv3ppS3kuk4kCZy1Lm67qvVDR47LtOoTZRU4NWyCv9ehBE6fXiRQRLLAYTgEa1+dYMF5BSNwpb1nyN+w8XczXR4Ch/5JuKVqPU609IF7//335WtTQxwEC7atKoGzTt9TUUGBK/+ojL5xFDjjfcYA1np+eYTqThRKlPXdifML99pR4NzjWuCUpHXbMECm/0o77yizEmkChwmBn3++rm1bJRP4kcaUPVaBU2PkQOQw6CXSW3fskkslcGr/mH8OQ2Lor1lFJAscBvFV7wkGVQ0WbBeMwL3UrJnt2tnKXmpm9l1DnfIQOHV79t//fly0aNHSFDL92IEikgVODb6q4vfff9erlAq2KUvgrPu3Rllf5KXFiJGjbNcA19U6GKm/QPkDluvvJiJZ4HDO8+bN07NdgXMqS+D061Ue102fVu3BB8q+bjhP6+fWTUSywK1Zs0Y+Ba6GdgoGNwL363r7HLUINYB1eQX2aR2E2230H/CebS5Vf6FeczhQ4NzjSuBCobwFLiXdGAi0ZN3+RWRdt6bxx/RKK2Ny34tXrjm2Q1y5FmvmqwFHVaCl4GpMyX8sgba/7As9X4VXBM4aGAzTDagbjMCV9R+8Gr1fvw6lhb+61lHmETHxSfJJZb1eWeElgVPhVsRR143ALS4eSLm84oNhw+V+9Xy34W9qLX8RqQL3ySef2K4XhiYJBmzjRuBW/tf959JNvDfQaPHV892GPnB6oIhkgcMTqNZrp8Z6dIMbgUP5n6f+suV9u7Rk5pLyCBzj9Nnzjvyyol//AVUqcKkJuaJ/04tiwEsX5TrSiFtZ+aIgv8hcr054RuBCDfwxKYGrqvCiwKn4+efSRx1HnWAEzmvhRYFTgZYPTMcVCNQJR+CwLQa/Vuv7Dx0VP65aI9Oz5nwsJXr9b5tky82uPfvNerrAbduxyzbsC6b4mTxlqhgxcrRMG/s+IrZu3+WLnXKKJ/m6vl1aqqR4ReCsgS4NZYF64QjcjJmzzfcVgWuz9udfzTL8Q/TLug3yuu09UHJddIH7/Y+ttlk08M/XpMkfiZGjPzSnXdu994Dc/+Yt28Tn8740r5v6O/EXXhI4FWq2nNJwK3B6norxfvpkj/5wjFzOmDVbLr9btlxeNzWDyX8WfiPGT5hkm8sYx4DAYWSFUb7t8VCZdZ9otMADZZMmTxFnzl0086ta4JSkfTXGGAj7q7EJNmEb+PIlMbZ7yWwp1YGoF7iXW7Qo9cuiMsLLAqfi2LFj+qYSlFHgqoayBM4a/kB+OALXoGFDWW6t+9HU6WZaheqC0KdPX1mmCxy6NjR9qeRBI7WdmsoHeU2aNhXPPvuc6PNuX9u+Maaa/rpUeFHgVJQGysMRuOfrGl1JrHU/+3yemVbx+JPGw0ZDhn4gy3SBw+wGqhsDRE1tpwavRn7t2nXk9X39DWMqJhX6FEzW8KLAWSMQbgWudu3ajnxVNmHiJHN96fIV5v701+AvrPvRY9KkKbJMDZVljfETJsqyqhY44iTqBS4Swp/AYeor/YPihcA8rFaQV90ETn9PvBKPPvqo4zzcCBz6OUGmVLzySkmLdmvfD7iqs3DREtt23Xv0NNevXI+VeUiXJnCNfT+Q48eX/EipUAKHdLTdQi0tunTpom8u890IHPrvWq9b+w4dzPIWxYOU47phmjrrdtY5ZtXg5UiXJnDI//iTzxyvQwkc0tF4C7W0WLBggW1bNwKHmOiTKet+XuvWXeZ/NHWabXukV6/9yUxP8ZVby75d+r1Mq77ep86cM8s+/exzs64++gJmvVFlu/cZYo5WWQpc5EGBq4TwJ3CYjB5zYwYb1g92VQRGMreCvOomcHj6szxjxowZIQXmZtSvT6DAnLeZmZm280C+G4EL1AJnrYPQ8/YdPOzIww9BaQKH/GN/nnIcozwETn9PvBRWsO5G4AK1wFnrIPS8sxcumevqxx/LsgROzRFtjfISOP398EpYp99yK3DWmDlrjtwGXQiwjjRub6u0qoc03l/runWuaKwfPnbCTJ84fdYsQ1cH5OE2K5a/rv/N9hqQd9JXnwIXeVDgKiH8CVyk4PYWaps2bfRNJSirbgIXKeCa6NdJD7TCXLzov2MvysMVuEFDhprH2rl7r207vUUGeViWJnCYJNzaIqSiPAQOwoGnxcsrMHh4KIGpm/TrFCjQj3HatGmWq1Y+Ate3X3/zGPss/dywPn/BN+a6eoob6bIE7qdf1jmOU14Cp7/34QQe8tGvidtQt4fdBL47/vzzT/McQhE4BLb5pLjFDP25sT5t+kzRvXsPW52/L1y2rZcmcGt/LrlW6OeIPCXrYydMMMvwcB7yrl6PkwL30EMUuEiCAlcJ4WWBw2TQpYE6FLiqoSyBu3btmr6JDdQJR+A2/bFVlmNw5AuXr8m0GmNKvQb11DF+NLGONP7DV2mEVeDQCds67MS8L7+WS6vAqf3rr0cPXeAihfXr1zuulb9455139E0lKAtH4PDAAsp37tknzp67KNPqoQN1bPVkt7Uf4vIffjTTCKvAoS+idUiRz+YafeqsAqf2r78ePfwJXKTw9ttvO66Tvzh/3hhmy0pZAgeBwvBXGC1B5Y0cNVpu83dx6xoePlHH0EdbCEbgMKyWKsO1U8M3YdYalKvrr1r5kcaDEvpx9VCvLRwocO6JeIFDvvWDgXj66WfMHwY8lab+wEKNmbPniIHvD3LkBxuBXofXBA7X4+pVd0/zoH4ggdP3i6hRo6Y5xAeGXwn0nrkNtNjUr9/AkR9MnLf089HDawI3atQovVpAUL8sgYNY6cdQ7xWWR46fNOt+8dV8WxkGylZ9hjp07GTbb8/iOWpr16kjevV+UwweMtQs+9T346+OgzTyhnwwTLzxRsl3BJ6gs74Wf+FFgevRo4de3QHqlSVwLzZq5Ni3EmOkj58sGapizsefmu8jlhd9Mt6gQQOZ7vraa7b9on8U8jH2Jq7puOIO7ojZH5f07cPfAvIwdV3//u+ZdZZ8t8yso79mFV4VuMmTJ+vVbZQlcIi1v6yzjVfarXsPhzA9Z5EqFVi3DqeE9fOXSgYCxrp6EhXp6/GJQs2I826//rZ94SlhdXy0llvL2rU3PlOqX54eartwoMC5J+IFrs2rr4pVq9ea63hUXf2RYB0j9+t/zMHGhEmTRZ++xhNy4USg1+EVgUMn97i4OL1KqWC70gRuzsclggBxs85w4faWSmmBAXrr1qvnyA8m8N9toNfhFYELZjwqBbYrS+BCDewbAqfnV2Z4SeDQCpKVlaVX9QvqlyVwoQb2DYHT8yszvCZweGrUDW4Ezk1gHxjuRc+PhFDvSThQ4NzjOYFDLFrynflBoMCFx5EjR+RrK+t2WyCwrVuBQ+B2m3qfKHDhkZqaKqZMmSIKCwv1IgdF+fl6ljwvClzlowtcRkaGXqVUsA0FrmqwClyrVq304lIpD4H7bdPmsPdRkaHem3CgwLnHkwJXq1Zt84/Yn8C1Lf7iVqEGNUT07Pm6mY8+G2h2tgqc6sip5mr8bdMftn1Zm7PxhJHKx60C/XWoiGSBCxecVzACp35UkfYncPo4X9ayoR8MM/NxOwiDTFoFDk9qoezP4ttDgwYPMetDxPQ+IyqWr1jpOJaKSBY4tyT37i1inn1WZGlTN+G8KkrgrP1xqioiVeAAngq+dOmSnu0KnFNFCZz1ScaqikgWOPRtQ7/g7GxjLvBgKA+Bw3fYteLZaiIx1HdqOFDg3OMJgbP+2CJwq0/1gdMFrkbNmrLjrVpH3xpVjmE4/M3dpwRO/XejPiBHjp2w7VvJHdLoY4LbgaoMna8DfTirs8D5C9UHThc4zJP6z+LOtAiMT6QG/Gzbrp3f91cJ3IFDR2X5+eIJoNGqhifGVD3r30mnzl1816+xZR/GU4H6vhFeFbiC1FQpbdbIPXfOVgfnVVECFwkRyQIXDjinihK4SIhIFrhwKA+Bi/RQ3/HhQIFzjycEztoCpyaZVx8EXeCQXvDNIscf1cUr1+XSOkilCgjcY4/9y9h24WIzv0XLlrKl58MxY81Qx8ISrUX6cfR9I6qzwFlb4OKTUszZA7CuCxzSA94b6NiHWvobdR8Cp/rV4WEElT9u/AQ5U4C6bqO1a2dtjTv658mA186LApc6bJhD3hBCu9WK86LAeQ+cEwXOe1Dg3EGBc4/nBA4RU/woNdL+BO7rBQsdf1RomcFSf+IGAYHD49soHzFylJnfrFlz2WK34JvFZuApKrXP3m+/7TiOvm8EBc6Zj6U/gdOvjyrHEuMw6fuCwKFFFuV79h8084eNGCleadXadu0Q1n2qOH3mnCNPhZcE7tYPPzikzSZwGjgvCpz3wDlR4LwHBc4dFDj3eFLg1PQeSONH2/qhwG0zjE+k1tVj62qMHettTxXqFuqOXXtkXUz0i3xMRRLoA/fkk0+JRx4tuUW3/+DhgHUpcCV56Bul3ic1Argqe/a558Sjjz1qru/au188+KBxyxvjgPl7f9Ut1BOnz8jyA4ePyvwt23YGHDW8bt16okOnkmEt1DyNej2EFwSuKC9PJLRu5RA2a8TWr69vJs+LAuc9cE4UOO9BgXMHBc49ES9w7dt3MP8orKF+qK1/NNNnGY9W63XPXbQPcGgNiJ2/hxhGj/lQrmMiZ6yjJQ5LTMztb19PPfW0XOqvH1GdBc5f/Hf1Gkedt97p43cbf/vDbW0sd+3Zb3uI4UpMnMzv3tMYoVyNmq7qWweIte7rrbeMJ8v014+IdIGLqVXLIWv+Im3YUH1TeV6hCtyncz93XCuEdSyqqg4KnDMmTrbPs6lCH2usKoMC5z/0a6ZCr1eVoV5TOFDg3BPxAuc28ESidR0PIvibmw+BfnQYgVzPV5GQYnSyV4EnUv/6+7z5ZKo1jp88LUfHRtraB8sa1VXg3MbJv0rm5UOcu3BZtqjp9RAY8V9NyoxQD7OogEBYJQLpU2f8/x1gfj81c0CgpyYjWeAKb992iFqgKEhP1zeX5xWqwH3iEzh9cN5ICwqcMyBwaHHW8yMpKHD+A9seO+GcJziSggJXuUSNwEVyUOC8G5EscKAgM1PE1K7tEDY9/FGRAoeJ7Kd8NE2s27DJlr9l6w653LFzj5g6bbpMY3Ju/HO0aPG3YumyFTIPQ//8Z8E3Yruvnr5vt0GBc0ZZAocuKZOnTJV9i635mzZvkcttO3bJCdaRXu+rg24QCxctFjt375N5cb5/jr+ev0Ds2mOshxIUOP9RlsDhs/bR9Bmyi5E1f/1vm2QLK7oEbdu5Wxw4fMz3D+sl+U/rtJkzzX9e8Y/uzFmzA/4z6yYocJULBa4SggLn3Yh0gVMkFY/1Fij8UVEC9+HYcXLfL9Svb36hqzKk1fyaz/hel8qzRo/isRrRhxHLbj16Oo7hJihwzihN4AYXdxfB09tY6l0O1MNCqsuCft3UGI6q60LfAQMcx3ATFDj/gW0DCZy6BmpsUkx5ppchMMVZzeKhtqz5Y4o/sypi4pMcx3ATavtwoMC5hwJXCUGB8254ReCALm0VLXDWL3yEGn8RaWvdf/l+MOYvMIb2QdnWHbts5Xp96/r7gwY7yt0GBc4Z/vrA4fsXZfr7/FiNGmLZipVmGfqcWsut9dt1MN5rtY75ba0CGExQ4PyHft0QmOz+s8/nifoN7PNBo0y1pCFt7f4DgbO+DqSfeKLk7wnrgeY6LSvU6woHCpx7KHCVEBQ474ZXBE71h8ucM0vENWpUKQLnrwVu996SJ8RVfPHV1/LJX6T1Mn951nUlHPo2boIC54xALXDbdux2vM8zZs0WDRs2kmm9TM8bNfpD2/qgwUP9buMmKHD+A9v6a4Hr3fstMXxEyfBXqu7W7cY/SvoxIXCtW7cx1/Edh0HUrdu+3KKl4zhuAtuGe+0ocO6hwFVCUOC8G14ROF3UEtq1NfMS2rSx1CwB51XeAodbL/oPRt9+/U1p0Mv85VnXJ06iwOngnMpb4PAglv4+937zTXOwcr1Mz3MKnDGNnb6Nm6DA+Q9s60/ghg4bLnr1tv9+ou6BQ0fMtLVMClwbu8BhzFPrthQ4b0CBq4SgwHk3vCBwsXXqBGxlS2jbVhSkpenZEpxXeQscwvgBaCE7Tn82d55cV8NU6D8m/vKs6xQ4Jzin8hY4BPaLcTeRxoMK1vfd3zWw5lHgyqaiBE6VjRk3Xqa7dn1N9le0llnrUuCihzIF7nZutnhqaXNRVFQk15FutfYNmS4sKpTrGfduWDeRBCtw6sJHc0Qj+jlGa0QqyX37Snm7MX++XlQmOK9QBW7L9p1yjEQ9X4WaXxbx87r1Zr6/wZVRx7r+8EMldTAQt/XHKJigwDljw6bNUor1fBWYS1pdN8wNrfIfKuO6rf7pF/HIwyUDm8/94kv5N6Bv4yYocP4D2/obygqx9PsV5nV75BH75wWCZl1/vm5d25SFTZo0FW++VTKr0JNPPik+mmo8IR5slMf3JQXOPWUK3Nbre6SkZd27KdeRRoD0u5ky/cvFzdZNJMEI3MqVK80LH80RjejnGI3RuHFj/bQjgrv79jlunQYDzi1UgfNCUOC8GRQ474b6zgwHCpx7yhQ4cCfvrpm+l29/Y2/n3rGtK4IROC9QZ7nRj2je1jhx626+VkpI5aPkraigQC9yBQXOm1DgvAkFzh2BBG5cj2uif9OLAjcD794plOnp/WJl2dzh8XL9zq3Qvgu9iiuBCwUKHCEVR2KnTiG3vCkocN6EAudNKHDuCCRwnw41JA0U5BfJ9JJpSXJ97fw0uZ6fV2TdJOqhwLmEAkcihbyrV6W85V64oBcFBQXOm1DgvAkFzh2BBI44ocC5hAJHIoVw+r1ZocB5EwqcN6HAuYMC5x4KnEsocCQSUPJWkJWlFwUNBc6bUOC8CQXOHRQ491DgXEKBI1XN7TVryuXWqQJftG3athWjPxwblYHpgcL9MYlEcE6Yb1Q/32iJzl27RuV1UwKnn280BQWucqHAuYQCR6qSorw8KW9JPXvqRSGjvmyjPaIN/fyiNaKNjRs3Os4xGqNGjRr6qQcFBc49FDiXUOBIVVJe/d4IIdHL/w7bK5fqaU0vQoFzDwXOJRQ4UlWkDBwo5S1t7Fi9iBBCTChw1QsKnEsocKQqSCmeKitr7ly9iBBCbFDgqhcUOJdQ4EhVAHmLa9hAzyaEEAcUuOoFBc4lFDhS2cTWqcN+b4QQ11DgqhcUOJdQ4EhlkjZmjCFvhYV6ESGE+IUCV72gwLmEAkcqi9xz5/jUKSEkaChw1QsKnEsocKSyoLwRQkKBAle9oMC5hAJHKoM7mzZJebt35IheRAghpUKBq15UmMDlFuSJp5Y2j6oAq46kyA8Jg1He0bLP91LeevT62lHGYDAYbgJA4LwcaYl5Vp0gAagwgSOEBAdvnRJCCHELBY6QCCD2+eelvOWnpOhFhBBCiAMKHCFVTHKfPlLeMj76SC8ihBBC/EKBI6SKgbylvPeenk0IIYQEhAJHCCGEEOIxKHCEEEIIIR6DAkcIIYQQ4jEocIQQQgghHoMCR6KG8hpDLalHDz2rQri9erWeFZiiIpE1b56eSwghpJpCgSNRQygCV5CWpmeFtJ9gub12bVDHKbxxI6j6hBBCohsKHIksiopEbL16eq4rQhEcf9v4yytvcI7BHCexY8eg6hNCCIluKHAksigslKJyd88evaRMQhEcbJM6aLBI7NzZvHUayn6CJbZOnaCOg7rB1CeEEBLdUOBI1VFQIOKbN3clJ/cOHdKzROGdO7Z12z58Ipjcq5e4+e23JXkWci9dMo+L1q0bCxaIgqwsWRbotSA/LzZWxLdoIbI+/VTm5Z47J5K6dRN3Nm7UapdOTO3aIqVfP3M97/p1kfT66wH7xeHYobZMEkIIiT4ocKTKMMWtqMiWnzF5spHwSVhC27Yi4dVXjbq1ahn5vvpqWwhabIMGMtsqXkjnbNsml3eL5S9z1myRs2uXyP5prbGbvDyR2LmTuY1C7ef+yRMi5rnnbPlyv9u3i/zUFJH15ZdyPffvv23Hvn/6tFnXun18s2Yl+cXbgXsHDxqvc98+236SenS31c/ZscMsI4QQUr2hwJEqJam7ISmZ06aZeUpiiu7elemUvv1EUX6+mY9l1ldfyXR+YoItH8S+8IJMxzVsaOT5RLAwJ0cKUH5ysmztUthawS5elEu5baNGMtBSlvnJJ2Z+YXa2WV+KlU/Q1PHA/ZMnzXRBSopM39myRdzZutXMV7eJ08eNK9kPjukTUXM/Z8+a6bzi1sK4Jk2M7QkhhFR7KHCkyshPTTXkyidF6lYqwDLvyhWRFxMjBUphLU/q2VNkr1sn04ldujrK833yVHj7trltIDJnzjTT1u1jahnHzd60ycyPff55sy5Q+dZbuVmffSrPpSAjQ5Ynv/22iHvxRZH1+ecipX9/U96srXaqlc76eiGbKMdtXSxvLlpk1ieEEEIocKTKSO7VW0pJxkcfifQJE0pExSdtECCglgAtXaAoN1e2rsXWr29Ijk/WgBKc9EmTZBqtWBnTp5cqPjk7dxrC5ov0sWNlHtK4vapQ26NFDgJm5he3vuVduyZShw416uH2bq1aMn3vyBFz+/ykJPM46EMn84vlVOXnnj8v0sePN45//76Zn/XFF7JeQuvWxoEJIYRUeyhwpErJu3ZVPv2Z+v77sn9asASSM8hQQptXRYZPDHOLb40G4uaSxa5a6/xxY/58Ed+qlcicM0fe8g2Vm0uW+AStlciYOlUU3im5TUsIIYT4gwJHPAdapNDiBQIJHCGEEBLNUOCI50A/M3V7MXNWSR82QgghpLpAgSOepCAzk61vhBBCqi0UOOJ5knu/qWd5jqzPPnMMTEwIIYQEggJHPE8oLXFF9+45BhDWwZhx1qdRK5Lk3r1F5uzZejYhhBDiFwoc8TxK4KxDjvijID3dTMc3aSpu/fCDpdRJQWqqKLh5U8+uEBK7dBE3Fy7UswkhhBC/UOCI51ECh+W9w4e1UoPbq1aZY60BCBzGfStISxP58fFabYPCGzdEzratenaFgDHeMCYdIYQQ4gYKHKlyMAZbUUGBnu0aswUuM1MU3SsZiw0D/mIKLYUstwyQqyKuaVOzjpXCmzdF5owZxkpRkcg9e9ZefuuWbR0zRwB5nPx8W5kOxoyzvjZMk6W2J4QQQsqCAkeqFMxmcGfjRqP17NgxIw9iVTxxPQbjxbq8/WmZxB6B/mnoy6bqqhkL5Hbt2pXULZ6qSk2bVZiVJbI++USkjR5t1vcHBC59zBgR92Ijo/WuWODyE4z5VzGpvZJHkNC+vVy/e/CgbNmTrxGvz/IaQGLnznLwXwxgrOY3te6HEEIIKQsKHKkycNsQ4pL9yy9yiYcGJErUfGKGKbMU8U2bytuaIPPjj41tEhPNOUqVBKlWNsX9M2ds5QD93yBSpYEWNmyT6BMzK8hLaNPGfA3WfEifdR3ihknrpczl55tid2fTJpHgkziIpqpLCCGEuIUCR6oMKTw+IUvs1EnkXb9uK4OUoTzP0j8N6zf+8x+ZRuubLI+57pMkowUOsnT/r798cdqvECEv9+IFmb67Z4+IrVvPzPd3yxO3OeMaNTaOu2CBmY/1ewcPisSuXW23fvVjYl0JKNKYmB5zsya/+abcNmvuXFtdQgghxC0UOFJlpI0caUjYtWtyCA0pWGfOFIuWMX8ppEzdfkR+xrRpcom4f/q0mQ9S3x9o5t36/nuznirHftR+VesaIv7ll2WeDp5qzV67VqYT2rW17Qdp3EpN7NatJF+TMHVrF9w/c1akDBhgti5mb9gg536V+/FJanzLliJ94kTL1oQQQkhgKHCkSrl/6pRIaNVapI8ebdzq9AnO3QBPkuqCFC7ZP/8sJSoUbq9ZI1931pdfSpELBrT2JfXuLfvA4VYqIYQQEiwUOOIZylvgCCGEEK9CgSOeAQKHBxQIIYSQ6g4FjniK7N9+07MIIYSQagcFjhBCCCHEY1DgCCGEEEI8BgWOeIb0SZPkzAaRAh+qIIQQUlVQ4IhnSJ8wQQ7ZISMxUS8uNzCWHOTs3qFDepENKXCFhXIcOznVFyGEEFJJUOBIlYNJ5yFCpaEG3ZXx3HMitl49IQoKxN19+2S5nBe1eFYEf7MqBAtmiCjMuaNnSzBrhO311KolYl94Qa9GCCGEVBgUOFKlYDBbJUKYJxTEN2sml8jLmDzFSNeuLZL79BH5sXHmtqrOzSVL5DK2zvMi99Il261NyF7aiBEiPyXFPE5ihw5mudpWbu+TMH2YEohhSv/+oqB4e7xGtAAmvPqq7TiEEEJIZUKBI1XGre+XGxKE6aV8gnbvyBGZjzxMLZXwyis2ScqPjTWnwlIo+cI+AGZIiGvSRKZztm6VZeg3h2X62LEyP755c3Hnjz/M7SFpalt1PPO4xVNfye0nThTJb79t5FvrEEIIIZUMBY5UGRCgwqwsEe8TtdyrV235aaNHm2klZ7hlmrNjh5mPuUrjGjUSt1evVpuK7PXrDSnLz5fLOz6JU/VT+vUzKvn2Z51kXoFbuabAPfecuHfihMjZts04VnEfN2v9QGlCCCGkoqHAkSoDEmWdSD5nyxazxUuBdO7Zv40ViFfTpiJz5kyzTvxLL4mkbt3M+uhLhzIVSv6seYhb331n5se3aCHiGjeWaTV5fVLPnnLQYMhh4muvmbvXX1vW55+LxC5d5OsihBBCKgsKHKlSEtq1M6VK9k1TrW0BQL0kn1CpBxbQgleYna3VMkDrnEKJl2xJ046Rn5wsCjIzbXkBsTwgcWvpUrnftJEjLRUIIYSQiocCR6KS9NGjReHNm+a6teWMEEII8ToUOBKV6MKmrxNCCCFehgJHohJd2FIHDxbxTdhPjRBCSHRAgSOEEEII8RgUOEIIIYQQj0GBI4QQQgjxGP8f/VFuK9kUq7AAAAAASUVORK5CYII=>