import unittest
import numpy as np

import miga

class MIGATestCase(unittest.TestCase):
    def setUp(self):
        self.miga = miga.MIGA()
        self.msa = np.load("msa.npz")

    def test_platform(self):
        self.assertIn("CPU", self.miga.available_platforms)
        self.assertEqual(self.miga.platform, "CPU")

        with self.assertRaises(ValueError):
            self.miga.platform = "CRAZY"

        self.assertEqual(self.miga.platform, "CPU")

        # self.miga.platform = "GPU"
        # self.assertEqual(self.miga.platform, "GPU")

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
