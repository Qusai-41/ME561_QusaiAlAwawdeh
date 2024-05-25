import random


# Define the target function
def target_function(x):
    return x ** 2 + 2 * x + 1


# Define the function set and terminal set
function_set = ['+', '*', 'sin', 'cos']
terminal_set = ['x', 'constant']

# Define the GP parameters
population_size = 100
max_generations = 10

# Initialize the population with random programs
population = []

for _ in range(population_size):
    # Randomly generate a program tree
    program_tree = generate_random_tree(max_depth=5)
    population.append(program_tree)

# Evolutionary loop
for generation in range(max_generations):
    # Evaluate the fitness of each program in the population
    fitness_scores = [evaluate_fitness(program, target_function) for program in population]

    # Select programs for reproduction based on fitness
    selected_programs = tournament_selection(population, fitness_scores)

    # Crossover: Combine genetic material of selected programs
    new_generation = []
    for _ in range(population_size):
        parent1 = random.choice(selected_programs)
        parent2 = random.choice(selected_programs)
        offspring = crossover(parent1, parent2)
        new_generation.append(offspring)

    # Mutation: Introduce random changes to selected programs
    new_generation = [mutate(program) for program in new_generation]

    # Update the population with the new generation
    population = new_generation

# Display the best program found
best_program = max(population, key=lambda program: evaluate_fitness(program, target_function))
print("Best Program:", best_program)
print("Fitness Score:", evaluate_fitness(best_program, target_function))
