```python
def erf(self, x):
    return self._create_op("stablehlo.erf", [x])
```
