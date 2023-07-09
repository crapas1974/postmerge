import unittest
from foo import foo

class TestFoo(unittest.TestCase):
    def test_0(self):
        result = foo(0)
        self.assertEqual(result, 1)
    
    def test_1(self):
        result = foo(1)
        self.assertEqual(result, 0)

    def test_other(self):
        result = foo(2)
        self.assertLess(result, 0)

if __name__ == "__main__":
    unittest.main()