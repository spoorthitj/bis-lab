import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Michalewicz Function
# -------------------------------

def objective(x):
    m = 10
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) *
                   (np.sin(i * x**2 / np.pi))**(2 * m))


# -------------------------------
# 2. Levy Flight
# -------------------------------

def levy_flight():
    beta = 1.5
    sigma = (
        (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
        (np.math.gamma((1 + beta) / 2) * beta *
         2**((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)

    return u / (abs(v)**(1 / beta) + 1e-10)


# -------------------------------
# 3. Cuckoo Search Algorithm
# -------------------------------

def cuckoo_search():

    n = 15                 # Number of nests
    pa = 0.25              # Discovery probability
    iterations = 300

    lower = 0
    upper = np.pi
    dim = 2

    # Generate initial nests
    nests = np.random.uniform(lower, upper, (n, dim))

    fitness = np.array([objective(x) for x in nests])

    best_index = np.argmin(fitness)
    best = nests[best_index].copy()
    best_value = fitness[best_index]

    history = []

    # Main loop
    for t in range(iterations):

        # Generate a new cuckoo using Levy flight
        for i in range(n):

            new_nest = nests[i] + 0.01 * levy_flight() * (
                nests[i] - best
            ) + 0.1 * np.random.randn(dim)

            new_nest = np.clip(new_nest, lower, upper)

            new_fitness = objective(new_nest)

            # Select a random nest
            j = np.random.randint(n)

            if new_fitness < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fitness

        # Abandon some worse nests
        abandon = np.random.rand(n) < pa

        new_nests = np.random.uniform(
            lower, upper, (n, dim)
        )

        nests[abandon] = new_nests[abandon]

        fitness = np.array([objective(x) for x in nests])

        # Find the best nest
        best_index = np.argmin(fitness)

        if fitness[best_index] < best_value:
            best_value = fitness[best_index]
            best = nests[best_index].copy()

        history.append(best_value)

    return best, best_value, nests, history


# -------------------------------
# 4. Run Algorithm
# -------------------------------

best, best_value, nests, history = cuckoo_search()

print("Best Solution:", best)
print("Best Fitness:", best_value)


# -------------------------------
# 5. Prepare Graph Data
# -------------------------------

x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)

X, Y = np.meshgrid(x, y)

Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))


# -------------------------------
# 6. Graph 1: 3D Surface Plot
# -------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title("Michalewicz Function - 3D Surface")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Fitness")

plt.show()


# -------------------------------
# 7. Graph 2: Contour Plot
# -------------------------------

plt.figure(figsize=(8, 6))

plt.contourf(X, Y, Z, levels=40, cmap='viridis')
plt.colorbar(label="Fitness")

plt.scatter(
    nests[:, 0], nests[:, 1],
    color='red', label='Final Nests'
)

plt.scatter(
    best[0], best[1],
    color='yellow', edgecolors='black',
    s=150, marker='*', label='Best Solution'
)

plt.title("Cuckoo Search - Nest Locations")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# -------------------------------
# 8. Graph 3: Convergence Graph
# -------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history)

plt.title("Cuckoo Search Convergence")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness Value")

plt.grid(True)
plt.show()
