
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 1. DELIVERY LOCATIONS
# --------------------------------

locations = np.array([
    [2, 8],    # Warehouse
    [3, 6],    # A
    [5, 7],    # B
    [8, 8],    # C
    [7, 5],    # D
    [9, 3],    # E
    [5, 2],    # F
    [3, 3],    # G
    [1, 5]     # H
])

names = [
    "Warehouse", "A", "B", "C", "D",
    "E", "F", "G", "H"
]

n = len(locations)

# --------------------------------
# 2. CALCULATE DISTANCE MATRIX
# --------------------------------

distance = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance[i][j] = np.linalg.norm(
            locations[i] - locations[j]
        )

# Heuristic information
heuristic = 1 / (distance + 1e-10)

# --------------------------------
# 3. ACO PARAMETERS
# --------------------------------

num_ants = 30
iterations = 100

alpha = 1
beta = 3
rho = 0.5
Q = 100

# Initial pheromone
pheromone = np.ones((n, n))

best_route = None
best_distance = float("inf")

convergence = []

np.random.seed(42)

# --------------------------------
# 4. ANT COLONY OPTIMIZATION
# --------------------------------

for iteration in range(iterations):

    all_routes = []
    all_distances = []

    for ant in range(num_ants):

        # Every ant starts at warehouse (0)
        route = [0]
        visited = {0}

        # Visit all remaining locations
        while len(route) < n:

            current = route[-1]

            # Find unvisited locations
            allowed = [
                city for city in range(n)
                if city not in visited
            ]

            # Calculate transition probabilities
            probabilities = []

            for city in allowed:

                value = (
                    pheromone[current][city] ** alpha
                    * heuristic[current][city] ** beta
                )

                probabilities.append(value)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            # Select next location
            next_city = np.random.choice(
                allowed, p=probabilities
            )

            route.append(next_city)
            visited.add(next_city)

        # Return to warehouse
        route.append(0)

        # Calculate total route distance
        total = sum(
            distance[route[i]][route[i + 1]]
            for i in range(len(route) - 1)
        )

        all_routes.append(route)
        all_distances.append(total)

        # Update global best
        if total < best_distance:
            best_distance = total
            best_route = route.copy()

    # --------------------------------
    # 5. PHEROMONE EVAPORATION
    # --------------------------------

    pheromone *= (1 - rho)

    # --------------------------------
    # 6. PHEROMONE DEPOSITION
    # --------------------------------

    for route, total in zip(all_routes, all_distances):

        deposit = Q / total

        for i in range(len(route) - 1):

            a = route[i]
            b = route[i + 1]

            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

    # Store best-so-far distance
    convergence.append(best_distance)

    if (iteration + 1) % 10 == 0:
        print(
            f"Iteration {iteration + 1}: "
            f"Best Distance = {best_distance:.2f}"
        )

# --------------------------------
# 7. DISPLAY BEST ROUTE
# --------------------------------

print("\nBEST DELIVERY ROUTE")

for city in best_route:
    print(names[city], end=" -> ")

print("\n\nMinimum Distance:", round(best_distance, 2))

# --------------------------------
# 8. GRAPH 1: CONVERGENCE GRAPH
# --------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, iterations + 1),
    convergence,
    marker='o',
    markersize=3
)

plt.title("ACO - Route Optimization Convergence")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Distance Found")
plt.grid(True)

plt.show()

# --------------------------------
# 9. GRAPH 2: BEST DELIVERY ROUTE
# --------------------------------

plt.figure(figsize=(9, 7))

x = locations[:, 0]
y = locations[:, 1]

plt.scatter(x, y, s=100)

for i in range(n):
    plt.annotate(
        names[i],
        (x[i], y[i]),
        xytext=(5, 7),
        textcoords="offset points"
    )

# Draw the best route
for i in range(len(best_route) - 1):

    a = best_route[i]
    b = best_route[i + 1]

    plt.plot(
        [x[a], x[b]],
        [y[a], y[b]],
        linestyle='-'
    )

plt.title("Optimized Parcel Delivery Route using ACO")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)

plt.show()
