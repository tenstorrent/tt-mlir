# TTMetal Dialect Ops

Auto-generated reference of operations in the `TTMetal` dialect (10 ops).

## `ttmetal.create_buffer`

Create buffer op.

Create buffer operation.

When this buffer uses a virtual grid, `virtualGridInverseMapping` stores the
inverse affine map (physical to virtual grid coordinates) and
`virtualGridForwardMapping` stores the forward affine map (virtual to
physical grid coordinates).  Both are propagated from d2m.empty through
bufferization.

## `ttmetal.create_global_semaphore`

Create global semaphore op.

Create global semaphore operation

## `ttmetal.create_local_semaphore`

Create local semaphore op.

Create local semaphore operation

## `ttmetal.deallocate_buffer`

Deallocate buffer op.

Deallocate buffer operation

## `ttmetal.enqueue_program`

Enqueue program op.

Enqueue program operation

## `ttmetal.enqueue_read_buffer`

Enqueue read buffer op.

Enqueue read buffer operation

## `ttmetal.enqueue_write_buffer`

Enqueue write buffer op.

Enqueue write buffer operation

## `ttmetal.finish`

Finish op for command queue.

Global barrier op, used to wait for all commands on queue to finish.

## `ttmetal.mesh_shard`

Nd sharding or (partial) concat op

Nd sharding or (partial) concat op in D2M runtime.
ShardToFull: Nd sharding in host memory.
FullToshard: (partial) concat in host memory.

## `ttmetal.reset_global_semaphore`

Reset global semaphore op.

Reset global semaphore operation

