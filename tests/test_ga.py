import unittest
import numpy as np
from scipy.spatial.distance import cdist

import miga

TESTS_REPEAT = 5

class BaseGATestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        msa = np.load("data_files/msa.npz")
        self.seq_a = msa["seq_a"]
        self.seq_b = msa["seq_b"]

    def platform(self):
        raise NotImplementedError("platform method not implemented")

    def setUp(self):
        self.miga = miga.MIGA()
        self.miga.platform = self.platform()

    def test_platform(self):
        self.assertEqual(self.miga.platform, self.platform())

    def assert_consistent_genome(self, seq_num, pop_size):
        genome = self.miga.genome
        for i in range(pop_size):
            u, count = np.unique(genome[i, :], return_counts=True)
            self.assertFalse(np.any(count != 1))
            self.assertEqual(count.sum(), seq_num)
            self.assertTrue(genome[i, :].min() == 0)
            self.assertTrue(genome[i, :].max() == seq_num - 1)

    def test_fitness_calculation(self):
        data = np.load("data_files/fitness_data.npz")
        ref_genome = data["genome"]
        ref_fitness = data["fitness"]

        self.miga.set_msa(self.seq_a, self.seq_b)
        self.miga.minimize = False
        self.miga.genome = ref_genome

        self.miga.run(0)
        self.assertTrue(np.allclose(self.miga.fitness, ref_fitness))

    def test_sorting(self):
        self.miga.set_msa(self.seq_a, self.seq_b)
        self.miga.run(0)
        self.assertTrue(np.all(np.diff(self.miga.fitness) <= 0))

        self.miga.minimize = True
        self.miga.run(0)
        self.assertTrue(np.all(np.diff(self.miga.fitness) >= 0))

    def test_consistent_genome(self):
        self.miga.set_msa(self.seq_a, self.seq_b)
        seq_num = self.miga.seq_a.shape[0]

        self.miga.run(3)
        self.assert_consistent_genome(seq_num, self.miga.pop_size)

    def test_maximize_target(self):
        self.miga.minimize = False
        self.miga.set_msa(self.seq_a, self.seq_b)
        non_elite = int(self.miga.elite * self.miga.pop_size)

        self.miga.run(0)
        first_max = self.miga.fitness.max()
        old_fit = self.miga.fitness[:non_elite]

        for step in range(TESTS_REPEAT):
            self.miga.run(1)
            new_fit = self.miga.fitness[:non_elite]
            self.assertTrue(np.all(new_fit >= old_fit))
            old_fit = new_fit

        self.assertGreater(self.miga.fitness.max(), first_max, "\nThis test sometimes fails. Run it again")

    def test_minimize_target(self):
        self.miga.minimize = True
        self.miga.set_msa(self.seq_a, self.seq_b)
        non_elite = int(self.miga.elite * self.miga.pop_size)

        self.miga.run(0)
        first_min = self.miga.fitness.min()
        old_fit = self.miga.fitness[:non_elite]

        for step in range(TESTS_REPEAT):
            self.miga.run(1)
            new_fit = self.miga.fitness[:non_elite]
            self.assertTrue(np.all(new_fit <= old_fit))
            old_fit = new_fit

        self.assertLess(self.miga.fitness.min(), first_min, "\nThis test sometimes fails. Please run it again")

    def test_non_elite_mutation(self):
        self.miga.set_msa(self.seq_a, self.seq_b)
        non_elite = int(self.miga.elite * self.miga.pop_size)

        self.miga.run(0)
        old_gen = self.miga.genome

        for step in range(TESTS_REPEAT):
            self.miga.run(1)
            new_gen = self.miga.genome
            self.assertTrue(np.all(np.diag(cdist(old_gen, new_gen, "hamming"))[non_elite:] != 0))
            old_gen = new_gen

    def tearDown(self):
        pass


#######
# CPU #
#######
class CPUGATestCase(BaseGATestCase):
    def platform(self):
        return "CPU"


#######
# GPU #
#######
# class GPUGATestCase(BaseGATestCase):
#     def platform(self):
#         return "GPU"

# Delete base class to avoid running it
del BaseGATestCase

if __name__ == "__main__":
    unittest.main()
