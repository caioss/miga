import numpy as np
import miga

# Data files
SEQUENCE_A = "sequence_a.txt"    # First group encoded MSA.
SEQUENCE_B = "sequence_b.txt"    # Second group encoded MSA.
RESTART_NAME = "restart"         # Filename prefix of the restart files (Numpy compressed file containing
                                 # both the genome and fitness values).
INITIAL_VALUES = None            # Restart file (.npz) containing an arbitrary genome to be used as the
                                 # starting point. If it is None, the first genome will be randomly created.

# Mutual Information parameters
LAMBDA = 0.5                     # Mutual Information pseudo-counter.
Q = 21                           # Maximum number of symbols in the MSA.

# Genetic algorithm parameters
GENERATIONS = 1000               # Total number of generations.
RESTART_FREQ = 100               # Number of generations between each restart (keep this number high
                                 # when using GPU to run the calculations).
POP_SIZE = 20                    # Genetic algorithm population size.
MINIMIZE = False                 # Minimize (True) or maximize (False) MI between groups.
MUTATION_RATE = 0.01             # Genome mutation rate.
DEATH_RATE = 0.25                # Population death rate.
ELITE_RATE = 0.1                 # Population elite (mutation-proof best individuals) rate.
THREADS = 4                      # Number of parallel threads to be used on the CPU platform.
PLATFORM = "GPU"                 # Platform (hardware) where computation will run. Valid values are
                                 # "CPU", "GPU" or "SimpleGPU".

# Settings parameters
solver = miga.MIGA()
solver.set_msa(np.loadtxt(SEQUENCE_A), np.loadtxt(SEQUENCE_B))
solver.lambda_ = LAMBDA
solver.q = Q
solver.pop_size = POP_SIZE
solver.minimize = MINIMIZE
solver.mutation = MUTATION_RATE
solver.death = DEATH_RATE
solver.elite = ELITE_RATE
solver.threads = THREADS
solver.platform = PLATFORM

if INITIAL_VALUES is not None:
    solver.genome = np.load(INITIAL_VALUES)["genome"]

# Restart parameters
windows = GENERATIONS // RESTART_FREQ
remaining_steps = GENERATIONS % RESTART_FREQ

# Running calculation
for step in range(windows):
    solver.run(RESTART_FREQ)
    np.savez_compressed("{}.{}".format(RESTART_NAME, step), genome = solver.genome, fitness = solver.fitness)

if remaining_steps:
    solver.run(remaining_steps)
    np.savez_compressed("{}.{}".format(RESTART_NAME, step + 1), genome = solver.genome, fitness = solver.fitness)
