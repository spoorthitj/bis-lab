
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# GA-TDX-INSPIRED GENETIC ALGORITHM FOR OPTIMIZATION
# =========================================================

print("=" * 60)
print("GENETIC ALGORITHM FOR CONSTRAINED OPTIMIZATION")
print("GA-TDX-INSPIRED IMPLEMENTATION")
print("=" * 60)


# ---------------------------------------------------------
# STEP 1: USER INPUT
# ---------------------------------------------------------

n = int(input("Enter the number of variables: "))

if n < 1:
    raise ValueError("Number of variables must be at least 1.")

func = input(
    "Enter objective function (example: x1**2 + x2**2): "
).strip()

choice = input(
    "Enter optimization type (min/max): "
).strip().lower()

if choice not in ["min", "max"]:
    raise ValueError("Enter min or max.")

print("\nEnter the bounds for each variable.")

lower = []
upper = []

for i in range(n):
    lo = float(input(f"Lower bound for x{i+1}: "))
    hi = float(input(f"Upper bound for x{i+1}: "))

    if lo >= hi:
        raise ValueError(
            "Each lower bound must be less than its upper bound."
        )

    lower.append(lo)
    upper.append(hi)

lower = np.array(lower)
upper = np.array(upper)

num_constraints = int(
    input("\nEnter number of inequality constraints: ")
)

if num_constraints < 0:
    raise ValueError("Number of constraints cannot be negative.")

constraints = []

print("\nEnter each constraint in the form g(x) <= 0.")
print("Example: x1 + x2 - 5")

for i in range(num_constraints):
    expression = input(f"Constraint {i+1}: ").strip()
    constraints.append(expression)

pop_size = int(input("\nEnter population size (minimum 4): "))
generations = int(input("Enter number of generations: "))

if pop_size < 4 or generations < 1:
    raise ValueError(
        "Population must be at least 4 and generations at least 1."
    )

penalty_coefficient = float(
    input("Enter static penalty coefficient (example: 1000): ")
)

if penalty_coefficient < 0:
    raise ValueError("Penalty coefficient cannot be negative.")

mutation_rate = 0.2


# ---------------------------------------------------------
# STEP 2: EXPRESSION EVALUATION
# ---------------------------------------------------------

allowed_functions = {
    "abs": abs,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "pi": np.pi
}


def evaluate_expression(expression, solution):

    variables = {
        f"x{i+1}": float(solution[i])
        for i in range(n)
    }

    value = eval(
        expression,
        {"__builtins__": {}},
        {**allowed_functions, **variables}
    )

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            "Expression produced a non-finite value."
        )

    return value


def objective(solution):
    return evaluate_expression(func, solution)


def constraint_values(solution):
    return np.array([
        evaluate_expression(expression, solution)
        for expression in constraints
    ])


# ---------------------------------------------------------
# STEP 3: STATIC PENALTY FUNCTION
# Constraints must have the form g(x) <= 0.
# ---------------------------------------------------------

def penalized_objective(solution):

    raw_value = objective(solution)

    values = constraint_values(solution)

    violation = np.sum(np.maximum(0, values) ** 2)

    penalty = penalty_coefficient * violation

    if choice == "min":
        return raw_value + penalty

    return raw_value - penalty


# Convert penalized objective into a score to maximize

def fitness(solution):

    value = penalized_objective(solution)

    if choice == "min":
        return -value

    return value


def is_feasible(solution, tolerance=1e-6):

    values = constraint_values(solution)

    return bool(np.all(values <= tolerance))


# ---------------------------------------------------------
# STEP 4: INITIALIZE POPULATION
# ---------------------------------------------------------

np.random.seed(42)

population = np.random.uniform(
    lower, upper, size=(pop_size, n)
)

best_solution = None
best_score = -np.inf
best_history = []


# ---------------------------------------------------------
# STEP 5: GENETIC ALGORITHM
# ---------------------------------------------------------

def select_parent():

    indices = np.random.choice(
        pop_size,
        size=min(3, pop_size),
        replace=False
    )

    winner = indices[np.argmax(scores[indices])]

    return population[winner].copy()


for generation in range(generations):

    scores = np.array([
        fitness(individual)
        for individual in population
    ])

    # Elitism: preserve the best individual

    elite_index = np.argmax(scores)

    elite = population[elite_index].copy()

    elite_score = scores[elite_index]

    # Keep the best solution found so far

    if elite_score > best_score:

        best_score = elite_score

        best_solution = elite.copy()

    # Record best penalized objective value

    best_history.append(
        penalized_objective(best_solution)
    )

    new_population = [elite.copy()]

    # Generate offspring using two-direction crossover
    # and grouped mutation

    while len(new_population) < pop_size:

        parent1 = select_parent()
        parent2 = select_parent()

        # Two-direction crossover approximation

        alpha = np.random.uniform(0, 1)

        child1 = (
            alpha * parent1 +
            (1 - alpha) * parent2
        )

        child2 = (
            alpha * parent2 +
            (1 - alpha) * parent1
        )

        children = [child1, child2]

        for child in children:

            # Grouped mutation

            if np.random.random() < mutation_rate:

                group_size = np.random.randint(1, n + 1)

                selected = np.random.choice(
                    n,
                    size=group_size,
                    replace=False
                )

                child[selected] += np.random.normal(
                    loc=0,
                    scale=0.1 * (
                        upper[selected] - lower[selected]
                    ),
                    size=group_size
                )

            # Keep variables within bounds

            child = np.clip(child, lower, upper)

            new_population.append(child.copy())

            if len(new_population) >= pop_size:
                break

    population = np.array(new_population)


# ---------------------------------------------------------
# STEP 6: CHECK FINAL POPULATION
# ---------------------------------------------------------

final_scores = np.array([
    fitness(individual)
    for individual in population
])

final_index = np.argmax(final_scores)

if final_scores[final_index] > best_score:

    best_solution = population[final_index].copy()

    best_score = final_scores[final_index]


raw_best_value = objective(best_solution)

final_penalized_value = penalized_objective(best_solution)

final_constraints = constraint_values(best_solution)

feasible = is_feasible(best_solution)


# ---------------------------------------------------------
# STEP 7: DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS")
print("=" * 60)

print("Optimization type:", choice.upper())

for i, value in enumerate(best_solution):
    print(f"x{i+1} = {value:.6f}")

print(f"\nObjective function value = {raw_best_value:.6f}")

print(
    f"Penalized objective value = {final_penalized_value:.6f}"
)

print("\nConstraint results:")

if num_constraints == 0:

    print("No inequality constraints were specified.")

else:

    for i, value in enumerate(final_constraints):

        status = (
            "SATISFIED" if value <= 1e-6 else "VIOLATED"
        )

        print(f"g{i+1}(x) = {value:.6f} -> {status}")

print("\nFeasible solution:", "YES" if feasible else "NO")


# ---------------------------------------------------------
# STEP 8: CONVERGENCE GRAPH
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, generations + 1),
    best_history,
    linewidth=2,
    label="GA-TDX"
)

plt.xlabel("Generation")
plt.ylabel("Best Penalized Objective Value")

plt.title("GA-TDX-Inspired Algorithm Convergence")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
