import unittest
from ttmlir.dialect.ttir.ttirops import TTIR_ToLayoutOp

class TestDtypeAttribute(unittest.TestCase):
    def test_dtype_attribute(self):
        # Create an instance of TTIR_ToLayoutOp
        op = TTIR_ToLayoutOp()
        
        # Set the dtype attribute
        op.dtype = "f32"
        
        # Get the dtype attribute and assert it is set correctly
        self.assertEqual(op.dtype, "f32")

if __name__ == '__main__':
    unittest.main()

def test_dtype_attribute_acceptance(self):
    # Create an instance of TTIR_ToLayoutOp
    op = TTIR_ToLayoutOp()
    
    # Set the dtype attribute with a valid value
    op.dtype = "f32"
    
    # Get the dtype attribute and assert it is set correctly
    self.assertEqual(op.dtype, "f32")
    
    # Test setting an invalid dtype attribute
    with self.assertRaises(ValueError):
        op.dtype = "invalid_dtype"

def test_dtype_attribute_lowering(self):
    # Create an instance of TTIR_ToLayoutOp
    op = TTIR_ToLayoutOp()
    
    # Set the dtype attribute
    op.dtype = "f32"
    
    # Lower the dtype attribute through TTIR to TTNN
    lowered_op = lower_dtype_attribute(op)
    
    # Assert that the dtype attribute is preserved after lowering
    self.assertEqual(lowered_op.dtype, "f32")

if __name__ == '__main__':
    unittest.main()