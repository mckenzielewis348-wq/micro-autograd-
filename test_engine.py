import unittest
from engine import Value


class TestValueOps(unittest.TestCase):
    def test_addition_and_multiplication(self):
        a = Value(2.0)
        b = Value(2.5)  # Changed b to 2.5 so (2.0 * 2.5) + 2.0 = 7.0
        c = (a * b) + a
        c.backward()
        self.assertEqual(c.data, 7.0)
        self.assertEqual(a.grad, 3.5)  # dc/da = b + 1 = 3.5
        self.assertEqual(b.grad, 2.0)  # dc/db = a = 2.0

    def test_relu(self):
        a = Value(-2.0)
        b = a.relu()
        b.backward()
        self.assertEqual(b.data, 0.0)
        self.assertEqual(a.grad, 0.0)


if __name__ == "__main__":
    unittest.main()
