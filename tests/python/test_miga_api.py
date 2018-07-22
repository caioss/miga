import unittest
import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing

import miga

DEFAULT_MUTATION = 0.01
DEFAULT_ELITE = 0.1
DEFAULT_DEATH = 0.25
DEFAULT_Q = 21
DEFAULT_POP_SIZE = 20
DEFAULT_LAMBDA = 0.5
DEFAULT_THREADS = multiprocessing.cpu_count()

class MigaAPITestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        msa = np.load("data_files/msa.npz")
        self.seq_a = msa["seq_a"]
        self.seq_b = msa["seq_b"]

    def setUp(self):
        self.miga = miga.MIGA()

    def test_wrong_platform(self):
        platform = self.miga.platform

        with self.assertRaises(ValueError):
            self.miga.platform = "CRAZY"

        self.assertEqual(self.miga.platform, platform)

    def test_q(self):
        self.assertEqual(self.miga.q, DEFAULT_Q)

        with self.assertRaises(ValueError):
            self.miga.q = 0

        with self.assertRaises(ValueError):
            self.miga.q = -1

        self.miga.q = 20
        self.assertEqual(self.miga.q, 20)

    def test_pop_size(self):
        self.assertEqual(self.miga.pop_size, DEFAULT_POP_SIZE)

        with self.assertRaises(ValueError):
            self.miga.pop_size = -1

        with self.assertRaises(ValueError):
            self.miga.pop_size = 0

        self.miga.pop_size = 10
        self.assertEqual(self.miga.pop_size, 10)

    def test_genome_shape(self):
        self.assertTupleEqual(self.miga.genome.shape, (DEFAULT_POP_SIZE, 0))
        self.miga.pop_size = 10
        self.assertTupleEqual(self.miga.genome.shape, (10, 0))

        self.miga.set_msa(self.seq_a, self.seq_b)
        seq_num = self.miga.seq_a.shape[0]

        self.assertTupleEqual(self.miga.genome.shape, (10, seq_num))
        self.miga.pop_size = 20
        self.assertTupleEqual(self.miga.genome.shape, (20, seq_num))

    def assert_consistent_genome(self, seq_num, pop_size):
        genome = self.miga.genome
        for i in range(pop_size):
            u, count = np.unique(genome[i, :], return_counts=True)
            self.assertFalse(np.any(count != 1))
            self.assertEqual(count.sum(), seq_num)
            self.assertTrue(genome[i, :].min() == 0)
            self.assertTrue(genome[i, :].max() == seq_num - 1)

    def test_genome_resize_consistent_data(self):
        self.miga.set_msa(self.seq_a, self.seq_b)
        seq_num = self.miga.seq_a.shape[0]
        self.assert_consistent_genome(seq_num, DEFAULT_POP_SIZE)

        self.miga.pop_size = 10
        self.assert_consistent_genome(seq_num, 10)

        self.miga.pop_size = 15
        self.assert_consistent_genome(seq_num, 15)

    def test_genome_resize_data_copy(self):
        self.miga.set_msa(self.seq_a, self.seq_b)
        genome20 = self.miga.genome

        self.miga.pop_size = 10
        genome10 = self.miga.genome

        self.miga.pop_size = 15
        genome15 = self.miga.genome

        self.assertTrue(np.array_equal(genome10, genome20[:10, :]))
        self.assertTrue(np.array_equal(genome10, genome15[:10, :]))
        self.assertFalse(np.array_equal(genome15[10:15, :], genome20[10:15, :]))

    def test_genome_shuffle(self):
        self.miga.set_msa(self.seq_a, self.seq_b)

        indices = np.triu_indices(self.miga.pop_size, k=1)
        dist = cdist(self.miga.genome, self.miga.genome, "hamming")[indices].sum()
        self.assertGreater(dist, 0)

        self.miga.pop_size = 25
        indices = np.triu_indices(5, k=1)
        dist = cdist(self.miga.genome[20:, :], self.miga.genome[20:, :], "hamming")[indices].sum()
        self.assertGreater(dist, 0)

    def test_fitness(self):
        self.assertTrue(np.all(self.miga.fitness == 0.0))

    def test_fitness_shape(self):
        self.assertTupleEqual(self.miga.fitness.shape, (DEFAULT_POP_SIZE,))
        self.miga.pop_size = 10
        self.assertTupleEqual(self.miga.fitness.shape, (10,))

        self.miga.set_msa(self.seq_a, self.seq_b)
        self.assertTupleEqual(self.miga.fitness.shape, (10,))

        self.miga.pop_size = 20
        self.assertTupleEqual(self.miga.fitness.shape, (20,))

    def test_fitness_resize_data_copy(self):
        self.miga.set_msa(self.seq_a, self.seq_b)
        # Fitness is read-only. We have to run the GA here
        self.miga.run(0)
        fitness20 = self.miga.fitness

        self.miga.pop_size = 10
        fitness10 = self.miga.fitness

        self.miga.pop_size = 15
        fitness15 = self.miga.fitness

        self.assertTrue(np.array_equal(fitness10, fitness20[:10]))
        self.assertTrue(np.array_equal(fitness10, fitness15[:10]))
        self.assertTrue(np.all(fitness15[10:15] == 0.0))

    def test_msa_input(self):
        self.assertTupleEqual(self.miga.seq_a.shape, (0, 0))
        self.assertTupleEqual(self.miga.seq_b.shape, (0, 0))

        seq_a = self.seq_a.copy()
        seq_b = self.seq_b.copy()

        with self.assertRaises(ValueError):
            self.miga.set_msa(self.seq_a, self.seq_b[:-1, :])

        with self.assertRaises(ValueError):
            seq_a[0] = -1
            self.miga.set_msa(seq_a, self.seq_b)

        with self.assertRaises(ValueError):
            seq_a[0] = self.miga.q
            self.miga.set_msa(seq_a, self.seq_b)

        with self.assertRaises(ValueError):
            seq_b[0] = -1
            self.miga.set_msa(self.seq_a, seq_b)

        with self.assertRaises(ValueError):
            seq_b[0] = self.miga.q
            self.miga.set_msa(self.seq_a, seq_b)

        with self.assertRaises(ValueError):
            self.miga.set_msa(self.seq_a, self.seq_b[:-1, :])

        self.miga.set_msa(self.seq_a, self.seq_b)

        self.assertTrue(np.array_equal(self.miga.seq_a, self.seq_a))
        self.assertTrue(np.array_equal(self.miga.seq_b, self.seq_b))

        self.assertEqual(self.miga.genome.shape[1], self.seq_a.shape[0])

    def test_threads(self):
        self.assertEqual(self.miga.threads, DEFAULT_THREADS)

        with self.assertRaises(ValueError):
            self.miga.threads = -1

        with self.assertRaises(ValueError):
            self.miga.threads = 0

        self.miga.threads = 2
        self.assertEqual(self.miga.threads, 2)

    def test_mutation(self):
        self.assertEqual(self.miga.mutation, DEFAULT_MUTATION)

        with self.assertRaises(ValueError):
            self.miga.mutation = -1

        with self.assertRaises(ValueError):
            self.miga.mutation = 0

        self.miga.mutation = 0.2
        self.assertEqual(self.miga.mutation, 0.2)

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

    def test_lambda(self):
        self.assertEqual(self.miga.lambda_, DEFAULT_LAMBDA)

        with self.assertRaises(ValueError):
            self.miga.lambda_ = -1

        self.miga.lambda_ = 0.1
        self.assertEqual(self.miga.lambda_, 0.1)

    def test_minimize(self):
        self.assertFalse(self.miga.minimize)

        self.miga.minimize = True
        self.assertTrue(self.miga.minimize)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
