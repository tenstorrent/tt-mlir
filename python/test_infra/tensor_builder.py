# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Union, Tuple

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir, tensor


class TensorBuilder:
    def __init__(self, ctx: Context, location: Location):
        self.ctx = ctx
        self.loc = location
        tt.register_dialect(self.ctx)
        ttir.register_dialect(self.ctx)

    @property
    def __default_dtype(self) -> Type:
        return F32Type.get(self.ctx)

    def ranked_tensor(
        self,
        shape: Union[List[int], Tuple[int]],
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        dtype = data_type if data_type is not None else self.__default_dtype
        with self.ctx, self.loc:
            return RankedTensorType.get(shape, dtype, encoding)

    def empty_tensor(
        self,
        shape: Union[List[int], Tuple[int]],
        data_type: Optional[Type] = None,
    ) -> Value:
        dtype = data_type if data_type is not None else self.__default_dtype
        return tensor.empty(self.ranked_tensor(shape, dtype), [])

    def empty_tensors(
        self,
        shapes: List[Union[List[int], Tuple[int]]],
        data_types: Optional[List[Type]] = None,
    ) -> List[Value]:
        if data_types:
            assert len(shapes) == len(data_types)
            return [
                self.empty_tensor(shape, data_type)
                for shape, data_type in zip(shapes, data_types)
            ]
        else:
            return [self.empty_tensor(shape) for shape in shapes]
