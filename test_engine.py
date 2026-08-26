import unittest
from engine import Value

class TestValueOps(unittest.TestCase):
    def test_addition_and_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a
        c.backward()
        self.assertEqual(c.data, 7.0)
        self.assertEqual(a.grad, 4.0) # dc/da = b + 1 = 4
        self.assertEqual(b.grad, 2.0) # dc/db = a = 2

    def test_relu(self):
        a = Value(-2.0)
        b = a.relu()
        b.backward()
        self.assertEqual(b.data, 0.0)
        self.assertEqual(a.grad, 0.0)

if __name__ == "__main__":
    unittest.main()
