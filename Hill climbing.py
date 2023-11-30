import random

def objective_function(x):
    # Replace this with the actual objective function you want to minimize
    return x**2  # Minimize a quadratic function

def hill_climbing(max_iterations=100):
    current_solution = random.uniform(-10, 10)  # Start with a random solution
    for _ in range(max_iterations):
        current_value = objective_function(current_solution)
        neighbor = current_solution + random.uniform(-0.5, 0.5)  # Small random change

        # Evaluate the neighbor
        neighbor_value = objective_function(neighbor)

        # Move to the neighbor if it's better
        if neighbor_value < current_value:
            current_solution = neighbor

    return current_solution, objective_function(current_solution)

# Example usage
best_solution, best_value = hill_climbing()
print(f"Best solution: {best_solution}")
print(f"Objective function value at best solution: {best_value}")
