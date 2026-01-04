import torch
import numpy as np
print("Testing dependencies...")
try:
    import torch
    import numpy as np
    print("SUCCESS: torch and numpy are available.")
except ImportError as e:
    print(f"FAILURE: {e}")
    exit(1)
