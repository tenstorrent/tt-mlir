```python
def iota(shape, dtype):
    import tensorflow as tf
    return tf.range(tf.math.reduce_prod(shape), dtype=dtype).reshape(shape)
```
