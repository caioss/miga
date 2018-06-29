import unittest
import numpy as np
import multiprocessing

import miga

DEFAULT_MUTATION = 0.01
DEFAULT_ELITE = 0.1
DEFAULT_DEATH = 0.25
DEFAULT_Q = 21
DEFAULT_POP_SIZE = 20
DEFAULT_LAMBDA = 0.5
DEFAULT_THREADS = multiprocessing.cpu_count()

class MIGATestCase(unittest.TestCase):
    def setUp(self):
        self.miga = miga.MIGA()
        self.msa = np.load("msa.npz")

    def test_wrong_platform(self):
        platform = self.miga.platform

        with self.assertRaises(ValueError):
            self.miga.platform = "CRAZY"

        self.assertEqual(self.miga.platform, platform)

    def test_cpu_platform(self):
        self.miga.platform = "CPU"
        self.assertEqual(self.miga.platform, "CPU")

    # TODO
    # def test_gpu_platform(self):
    #     self.miga.platform = "GPU"
    #     self.assertEqual(self.miga.platform, "GPU")

    def test_q(self):
        self.assertEqual(self.miga.q, DEFAULT_Q)

        with self.assertRaises(ValueError):
            self.miga.q = 0

        with self.assertRaises(ValueError):
            self.miga.q = -1

        self.miga.q = 20
        self.assertEqual(self.miga.q, 20)

        self.miga.q = DEFAULT_Q

    def test_pop_size(self):
        self.assertEqual(self.miga.pop_size, DEFAULT_POP_SIZE)

        with self.assertRaises(ValueError):
            self.miga.pop_size = -1

        with self.assertRaises(ValueError):
            self.miga.pop_size = 0

        self.miga.pop_size = 10
        self.assertEqual(self.miga.pop_size, 10)

        self.miga.pop_size = DEFAULT_POP_SIZE

    def test_threads(self):
        self.assertEqual(self.miga.threads, DEFAULT_THREADS)

        with self.assertRaises(ValueError):
            self.miga.threads = -1

        with self.assertRaises(ValueError):
            self.miga.threads = 0

        self.miga.threads = 2
        self.assertEqual(self.miga.threads, 2)

        self.miga.threads = DEFAULT_THREADS

    def test_mutation(self):
        self.assertEqual(self.miga.mutation, DEFAULT_MUTATION)

        with self.assertRaises(ValueError):
            self.miga.mutation = -1

        with self.assertRaises(ValueError):
            self.miga.mutation = 0

        self.miga.mutation = 0.2
        self.assertEqual(self.miga.mutation, 0.2)

        self.miga.mutation = DEFAULT_MUTATION

    def test_death(self):
        self.assertEqual(self.miga.death, DEFAULT_DEATH)

        with self.assertRaises(ValueError):
            self.miga.death = -1

        with self.assertRaises(ValueError):
            self.miga.death = 1

        with self.assertRaises(ValueError):
            self.miga.death = 1.01 - DEFAULT_ELITE

        self.miga.death = 0.2
        self.assertEqual(self.miga.death, 0.2)

        self.miga.death = DEFAULT_DEATH

    def test_elite(self):
        self.assertEqual(self.miga.elite, DEFAULT_ELITE)

        with self.assertRaises(ValueError):
            self.miga.elite = -1

        with self.assertRaises(ValueError):
            self.miga.elite = 1

        with self.assertRaises(ValueError):
            self.miga.elite = 1.01 - DEFAULT_DEATH

        self.miga.elite = 0.2
        self.assertEqual(self.miga.elite, 0.2)

        self.miga.elite = DEFAULT_ELITE

    def test_lambda(self):
        self.assertEqual(self.miga.lambda_, DEFAULT_LAMBDA)

        with self.assertRaises(ValueError):
            self.miga.lambda_ = -1

        self.miga.lambda_ = 0.1
        self.assertEqual(self.miga.lambda_, 0.1)

        self.miga.lambda_ = DEFAULT_LAMBDA

    def test_minimize(self):
        self.assertFalse(self.miga.minimize)
        self.miga.minimize = True
        self.assertTrue(self.miga.minimize)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
