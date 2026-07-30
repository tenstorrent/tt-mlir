# While Op Buffer Lifetime

How `ttnn.while` keeps the values it carries alive across the nested programs
that implement its regions, and a known gap in that protection when
`TTNNForceFinalDeallocs` promotes an aliasing deallocation to `force=true`.

## Motivation

Each region of a `ttnn.while` is serialized as its own flatbuffer program and
executed by a nested `ProgramExecutor`. The loop hands the condition and body
programs the values it carries, and those programs must not free them: the body
needs its arguments for the whole iteration, the captures are read by every
iteration, and the values yielded by iteration N are the inputs of iteration
N+1.

Three separate mechanisms decide whether a buffer survives, and it is worth
being precise about which one does what, because they do not compose the way one
might assume.

### 1. `retain`, checked before deallocation is even attempted

`retain` lives on `TTNNTensorWrapper`, not on the metal tensor
(`runtime/include/tt/runtime/detail/ttnn/types/types.h`). The runtime's
deallocate op consults it first
(`runtime/lib/ttnn/operations/deletion/deallocate.cpp`):

```cpp
if (!tensorWrapper.shouldRetain()) {
  ::ttnn::deallocate(ttnnTensor, op->force());
}
tensorPool.erase(op->in());
```

Because the check happens before `::ttnn::deallocate` is called at all, `retain`
suppresses a deallocation regardless of the `force` flag. This is the strongest
of the three mechanisms, and it is the one const-eval and trace already use to
protect tensors that cross a program boundary.

### 2. Buffer refcounting, which `force=false` respects

`TTNNTensorWrapper` holds its `::ttnn::Tensor` **by value**, and that is itself a
refcounted handle to the buffer. `Tensor::deallocate_impl`
(`ttnn/core/tensor/tensor.cpp`) declines to free a shared buffer unless forced:

```cpp
auto can_deallocate = [](const std::shared_ptr<T>& shared_resource, bool force) {
    return shared_resource.use_count() == 1 || (use_count() > 1 && force);
};
...
[force](DeviceStorage& storage) {
    if (!storage.is_sole_owner_of_device_memory() && !force) { return; }
    storage.deallocate();
}
```

with `is_sole_owner_of_device_memory()` being
`mesh_tensor_holder_.use_count() == 1 && get_root_mesh_tensor().use_count() == 1`
(`ttnn/core/tensor/storage.cpp`).

So merely holding a second `::ttnn::Tensor` handle makes every `force=false`
deallocation of that buffer a no-op.

### 3. `force=true`, which defeats refcounting on purpose

`TTNNForceFinalDeallocs` exists precisely because mechanism 2 is too effective.
Its header comment explains that when several handles alias one buffer every
`force=false` deallocation becomes a no-op, the buffer is never freed, and "this
can result in L1 allocation failure". The pass therefore walks each buffer's
deallocations and promotes the last one in program order to `force=true`,
erasing the rest.

### How the loop uses these

`runtime/lib/ttnn/operations/control_flow/while_op.cpp` builds each nested
program's input vector out of *private retained views*: a fresh
`TTNNTensorWrapper` over the same `::ttnn::Tensor`, created with `retain=true`
via `createRuntimeTensorFromTTNN`, its version aligned with `syncVersion`. No
data is copied.

This is deliberately a view rather than a flag flip on the caller's wrapper.
`retain` is not the loop's to modify: const-eval keeps its cached outputs
retained, trace keeps its input and output slots retained, and a host caller can
retain a tensor it passes in as a program argument. When the loop bounds are
const-eval'd — the common case — every init and most captures arrive already
retained and registered in `GlobalTensorCache`, so clearing their flag on loop
exit would let a later deallocation free a cached tensor, with the damage
surfacing only on the *next* invocation.

The views live exactly as long as the sub-program call. Mechanism 1 protects
against `force=false` and `force=true` alike; mechanism 2 comes along for free
as a second line of defence.

## Proposed Changes

### The gap

`retain` is per **wrapper**, but a buffer is shared. A `force=true` deallocation
issued through *some other* wrapper frees the buffer without ever consulting the
loop's retain flag.

`TTNNForceFinalDeallocs` can issue exactly that inside a while region. The
relevant details of the pass:

- It collects deallocations with `funcOp.walk`, which is **recursive**, so
  deallocations inside while regions are in scope.
- `canonicalRoot` follows view-alias chains. Today `getViewSource` recognizes
  only view-eligible reshapes (`canReshapeBeView`).
- `collectDoNotForceRoots` exempts values reaching a `func::ReturnOp`, and conv
  activations the conv frees itself. It knows nothing about `ttnn.yield`, and
  nothing about a region's block arguments.
- Forcing only happens when a root has **two or more** aliasing deallocations
  (`deallocCountByRoot.lookup(root) < 2` continues without forcing).

That last point makes the gap much narrower than it first appears. A while
region's block arguments never receive a deallocation of their own —
`TTNNDeallocate` only considers function arguments and op results, and
`checkAndInsertDeallocation` skips values whose last use is a terminator. So a
body containing a *single* view of a carried value gives that root a count of 1
and nothing is forced.

The reachable shape is therefore:

> a buffer whose canonical root is defined outside the region (a while region
> block argument, i.e. a loop-carried value or a capture) and which has **two or
> more** aliasing deallocations *inside* that region.

Then one of those deallocations is promoted to `force=true`, and on the first
iteration it frees a buffer the loop still needs. Concretely: a body that takes
two view-eligible reshapes of the same block argument.

A second, milder problem sits alongside it. The pass picks "the last in program
order" by reverse-iterating `funcOp.walk` order. For a body-local temporary that
is harmless — forcing its deallocation inside the body frees it once per
iteration, which is correct. But "program order" is not meaningful for an op that
runs N times, so the notion the pass relies on does not extend into a loop body
in general.

### Fix

Extend `collectDoNotForceRoots` in
`lib/Dialect/TTNN/Transforms/TTNNForceFinalDeallocs.cpp` so a root that escapes
its region is never forced, mirroring how values escaping the function are
already handled:

- treat `ttnn.yield` operands as do-not-force roots, exactly as `func::ReturnOp`
  operands are today;
- treat the block arguments of any `ttnn.while` region as do-not-force roots,
  since the buffer is owned by the enclosing program or by the previous
  iteration.

Both are additions to an existing set, so the change is local and does not
disturb the pass's handling of straight-line code.

An alternative worth weighing is to run the pass per region rather than per
function, which would also make the program-order reasoning well founded instead
of merely unused. That is a larger change and should not be attempted without a
failing case in hand.

## Test Plan

A lit test under `test/ttmlir/Dialect/TTNN/while/` whose body takes two
view-eligible reshapes of the same loop-carried value, checking that neither
resulting `ttnn.deallocate` has `force = true`.

The precondition to satisfy first is `canReshapeBeView`, which is what decides
whether a reshape aliases its input at all. If it turns out that no reshape of a
loop-carried tensor can be view-eligible under the layouts `TTNNLayout` assigns
inside a while region, the gap is unreachable in practice and this document
should record that instead of the fix being made.

End-to-end confirmation needs silicon: run the resulting flatbuffer with `ttrt
run` and check the loop's results, since a premature free shows up as wrong
values or an allocation failure on iteration 2 rather than as a compile error.

## Concerns

- **This is a reasoned gap, not an observed failure.** It follows from reading
  `TTNNForceFinalDeallocs`, `Tensor::deallocate_impl` and the runtime deallocate
  op, and no failing case has been constructed. The `>= 2` deallocations
  precondition may well make it unreachable with today's view-op set, which is
  exactly one view op.
- **The runtime path is not silicon-verified at all.** At the time of writing
  there is no device available in the development environment and `ttrt` is not
  installed, so `ttnn.while` has been exercised only as far as flatbuffer
  translation. That applies to the retained-view protection generally, not just
  to this gap.
- **`getViewSource` is a single point of truth that will grow.** Its comment
  anticipates a `ViewOpInterface` replacing the body. Every op added there
  widens this gap, because it creates more ways for a body to produce two
  aliasing deallocations of a carried value. Whoever adds the second view op
  should revisit this.
- **`meshEvent` diverges between a view and its source.** The view copies the
  event at construction and the two are independent thereafter. This is
  harmless while the field is mostly unset, as its own comment says, but it
  becomes a hazard once non-blocking readbacks and multiple command queues are
  used more widely.
