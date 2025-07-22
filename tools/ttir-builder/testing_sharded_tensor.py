# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from sharded_tensor import ShardedTensor
import torch

# Generate 8 shards of shape (10, 10) with constant values. Values are index of shards. Dtype : fp32
shards = [torch.full((2, 2), i, dtype=torch.float32) for i in range(8)]

shard_shape = (2, 4)

st = ShardedTensor(shards, shard_shape)

print("ShardedTensor itself")
print(st)
print(f"Shape: {st.shape}")
print(f"Dtype: {st.dtype}")
print(f"Stride: {st.stride()}")
print(f"Numel: {st.numel()}")
# print(f"DataPtr: {st.data_ptr()}")

# print(st.get_shard((0, 0)))
# print(st.get_shard((0, 1)))
# print(st.get_shard((1, 0)))
# print(st.get_shard((1, 1)))

# # Generate another ShardedTensor with same shape, dtype and values
# shards_r = [torch.full((2, 2), i, dtype=torch.float32) for i in range(8)]
# st_r = ShardedTensor(shards_r, shard_shape)

# #elwise add using torch.add
# sum = torch.add(st, st_r)
# print(type(sum))

# print("Sum of two ShardedTensors")
# print(sum)
# print(sum.get_shard((0, 0)))
# print(sum.get_shard((0, 1)))
# print(sum.get_shard((1, 0)))
# print(sum.get_shard((1, 1)))


# print("Concatenation of two ShardedTensors")
# concat = torch.cat([st, st_r], dim=0)
# print(concat)
# print(concat.get_shard((0, 0)))
# print(concat.get_shard((0, 1)))
# print(concat.get_shard((1, 0)))
# print(concat.get_shard((1, 1)))


# concat_cont = concat.contiguous()
# print("Contiguous ShardedTensor")
# print(concat_cont)
# print(concat_cont.get_shard((0, 0)))
# print(concat_cont.get_shard((0, 1)))
# print(concat_cont.get_shard((1, 0)))
# print(concat_cont.get_shard((1, 1)))
