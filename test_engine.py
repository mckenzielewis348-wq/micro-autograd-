def test_tanh_and_pow(self):
    x = Value(0.5)
    y = x.tanh() ** 2
    y.backward()
    self.assertAlmostEqual(y.data, 0.21378, places=4)
