import numpy as np

rng = np.random.default_rng(32)

goal = np.array([5, 2, 3])


def fitness(sol):
    return np.sum(np.square(goal - sol))


ngen = 10000
popsize = 50
sigma = 0.1  # noise standard deviation
alpha = 0.1  # learning rate

guess = rng.integers(low=-10000, high=10000, size=3)

for i in range(ngen):
    if i % (ngen // 10) == 0:
        print(f"gen:{i}, fitness:{fitness(guess):.3f}, guess:{guess}")
    N = rng.normal(size=(popsize, 3))  # sample from normal distribution
    F = np.zeros(popsize)
    for j in range(popsize):
        w = guess + sigma * N[j]
        F[j] = fitness(w)
    # standardize fitnesses to have guassian (normal) distribution
    A = (F - np.mean(F)) / np.std(F)

    guess = guess - alpha / (popsize * sigma) * np.dot(np.transpose(N), A)
