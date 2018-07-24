#include "MatData.hpp"
#include "test_tools.hpp"
#include "CPUPopulation.hpp"
#include "types.h"
#include "gtest/gtest.h"
#include <vector>

static const MatData<int32_t> GENOME("../data/shuffled_genome.bin");
static const MatData<int32_t> SEQ_A("../data/seq_a.transposed.bin");
static const MatData<int32_t> SEQ_B("../data/seq_b.transposed.bin");
static const MatData<int32_t> INDICES("../data/sorted_indices.bin");
static const MatData<double> FITNESS("../data/sorted_fitness.bin");

/*
 * Population typed fixture
 */
template <class T>
class GATest : public ::testing::Test
{
public:
    T *population;
    index_t pop_size;
    index_t elite;
    index_t survivors;
    seq_t q;
    double mutation;
    double lambda;
    bool minimize;
    std::vector<index_t> genome;
    std::vector<index_t> seq_a;
    std::vector<index_t> seq_b;
    std::vector<data_t> fitness;

    GATest()
    : population { new T() },
      pop_size { GENOME.shape[0] },
      elite { index_t(0.1 * pop_size) },
      survivors { index_t(pop_size - 0.25 * pop_size) }, // death ratio
      q { 21 },
      mutation { 0.01 },
      lambda { 0.5 },
      minimize { false },
      genome ( GENOME.data ),
      seq_a ( SEQ_A.data ),
      seq_b ( SEQ_B.data ),
      fitness ( FITNESS.data )
    {
        // Set data
        population->set_q(q);
        population->set_lambda(lambda);
        population->set_threads(4);
        population->set_msa(
            GENOME.shape[1],
            seq_a.data(),
            SEQ_A.shape[0],
            seq_b.data(),
            SEQ_B.shape[0]
        );
        population->set_genome(genome.data(), pop_size);
        population->set_fitness(fitness.data());
    }

    ~GATest()
    {
    }
};

using PopulationTypes = ::testing::Types<CPUPopulation>;

TYPED_TEST_CASE(GATest, PopulationTypes);

/*
 * Test if all matrices are consistent after
 * pre- and post-processing
 */
TYPED_TEST(GATest, InitializeFinalize)
{
    TypeParam *pop = this->population;

    pop->initialize();
    pop->finalize();

    EXPECT_TRUE(vectors_equal(this->genome.data(), GENOME.ptr(), GENOME.size()));
    EXPECT_TRUE(vectors_equal(this->seq_a.data(), SEQ_A.ptr(), SEQ_A.size()));
    EXPECT_TRUE(vectors_equal(this->seq_b.data(), SEQ_B.ptr(), SEQ_B.size()));

    for (size_t i = 0; i != FITNESS.size(); ++i)
    {
        EXPECT_EQ(this->fitness[i], FITNESS[i]);
    }
}

/*
 * Test the computed fitness values
 * Takes sorting effects into account
 */
TYPED_TEST(GATest, FitnessValue)
{
    TypeParam *pop = this->population;

    pop->initialize();
    pop->sort(false);
    pop->finalize();

    for (size_t i = 0; i != FITNESS.size(); ++i)
    {
        EXPECT_FLOAT_EQ(this->fitness[i], FITNESS[i]);
    }
}

/*
 * Test ascending and descending fitness sorting
 */
TYPED_TEST(GATest, Sorting)
{
    TypeParam *pop = this->population;
    auto &fitness = this->fitness;

    // Descending
    pop->initialize();
    pop->sort(false);
    pop->finalize();

    for (size_t i = 1; i != FITNESS.size(); ++i)
    {
        EXPECT_GE(fitness[i - 1], fitness[i]);
    }

    // Ascending
    pop->initialize();
    pop->sort(true);
    pop->finalize();

    for (size_t i = 1; i != FITNESS.size(); ++i)
    {
        EXPECT_LE(fitness[i - 1], fitness[i]);
    }
}

/*
 * Test genome contents after sorting
 */
TYPED_TEST(GATest, GenomeReorder)
{
    TypeParam *pop = this->population;
    auto *genome = this->genome.data();
    const index_t seq_num { GENOME.shape[1] };

    // Descending
    pop->initialize();
    pop->sort(false);
    pop->finalize();

    for (index_t i = 0; i != this->pop_size; ++i)
    {
        EXPECT_TRUE(vectors_equal(genome + i * seq_num, GENOME.ptr() + INDICES[i] * seq_num, seq_num));
    }
}

/*
 * Test if all new individuals are sons from survivors 1 and 2.
 * Survivor 0 must not reproduce.
 * Test if survivors genome aren't touched
 */
TYPED_TEST(GATest, Reproduce)
{
    const index_t pop_size { this->pop_size };
    const index_t seq_num { GENOME.shape[1] };

    TypeParam *pop = this->population;
    auto *genome = this->genome.data();

    pop->initialize();
    pop->kill_and_reproduce(3, pop_size - 1, 1, 3);
    pop->finalize();

    int second_sons { 0 };
    int third_sons { 0 };
    for (index_t killed = 3; killed != pop_size - 1; ++killed)
    {
        // New genome must be copies of individuals 1 or 2
        if (genome[killed * seq_num] == GENOME[seq_num])
        {
            ++second_sons;
            EXPECT_TRUE(vectors_equal(genome + killed * seq_num, GENOME.ptr() + seq_num, seq_num));
        }
        else if (genome[killed * seq_num] == GENOME[2 * seq_num])
        {
            ++third_sons;
            EXPECT_TRUE(vectors_equal(genome + killed * seq_num, GENOME.ptr() + 2 * seq_num, seq_num));
        }
        else
        {
            ADD_FAILURE() << "Son genome (index " << killed << ") mismatch with available parents (indices 1 and 2)";
        }
    }

    EXPECT_GT(second_sons, 0) << "Second parent had no children";
    EXPECT_GT(third_sons, 0) << "Third parent had no children";

    // Survivors testing
    for (index_t i = 0; i != 3; ++i)
    {
        EXPECT_TRUE(vectors_equal(genome, GENOME.ptr(), seq_num, i));
    }

    EXPECT_TRUE(vectors_equal(genome, GENOME.ptr(), seq_num, pop_size - 1));
}

/*
 * Test if mutation is applied only to certain individuals
 * and at the correct ratio.
 */
TYPED_TEST(GATest, Mutate)
{
    const index_t pop_size { this->pop_size };
    const index_t seq_num { GENOME.shape[1] };

    auto *genome { this->genome.data() };
    TypeParam *pop = this->population;

    pop->initialize();
    pop->mutate(this->mutation, 1, pop_size - 1);
    pop->finalize();

    // Mutated
    for (index_t i = 1; i != pop_size - 1; ++i)
    {
        double changes { 0 };
        for (index_t j = 0; j != seq_num; ++j)
        {
            if (genome[i * seq_num + j] != GENOME[i * seq_num + j])
            {
                ++changes;
            }
        }
        changes /= 2 * seq_num; // Swap is between two positions
        EXPECT_GT(changes, 0.0);
        EXPECT_LE(changes, this->mutation);
    }

    // Non mutated
    EXPECT_TRUE(vectors_equal(genome, GENOME.ptr(), seq_num));
    EXPECT_TRUE(vectors_equal(genome, GENOME.ptr(), seq_num, pop_size - 1));
}
