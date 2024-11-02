import numpy as np
import random

# Setup variables
selectionHand = [
    [0, 0, 0, 0, 12, 0, 24, 28, 38, 0], # Trynna get 29
    [4, 5, 8, 10, 12, 1, 24, 34, 1, 1],
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [1, 1, 1, 1, 15, 1, 24, 34, 11, 11],
    [1, 1, 1, 1, 15, 1, 24, 34, 1, 21],
    [4, 5, 8, 11, 15, 21, 24, 9, 2, 1]
]
battlefields = 10
population_size = 100
generations = 1000
max_Troops = 100
mutation_rate = 0.1

# Crossover function
def crossover(parents1, parents2):
    crossover_point = np.random.randint(1, battlefields - 1)
    child1 = np.concatenate((parents1[:crossover_point], parents2[crossover_point:]))
    child2 = np.concatenate((parents2[:crossover_point], parents1[crossover_point:]))
    return child1, child2

def mutate(setUp):
    if np.random.rand() < mutation_rate:
        mutation_index = np.random.randint(battlefields)
        
        # Calculate the maximum possible troop number that can be allocated to the mutation index
        max_allocation = max_Troops - sum(setUp) + setUp[mutation_index]
        
        if max_allocation > 0:
            setUp[mutation_index] = np.random.randint(0, max_allocation)
        else:
            setUp[mutation_index] = np.random.randint(0, setUp[mutation_index] + 1)  # Randomly reduce troops if over-allocation
        
    return rebalance(setUp)

def rebalance(setUp):
    total_troops = sum(setUp)

    # If troops exceed max_Troops, remove troops randomly from positive indices
    while total_troops > max_Troops:
        positive_indices = np.where(np.array(setUp) > 0)[0]  # Find indices with more than 0 troops
        if len(positive_indices) > 0:
            remove_from = np.random.choice(positive_indices)
            setUp[remove_from] -= 1
            total_troops -= 1
        else:
            break  # Safety check, but this case shouldn't occur

    # If troops are less than max_Troops, add troops randomly to any index
    while total_troops < max_Troops:
        add_to = np.random.randint(len(setUp))  # Randomly select any index to add a troop
        setUp[add_to] += 1
        total_troops += 1

    return setUp

# Fitness function: higher is better
def fitness(currentHand):
    total = 0
    for idx in range(len(selectionHand)):
        total += calculate_score(currentHand, selectionHand[idx])
    
    return total/len(selectionHand)

def selection(population):

    tournament_size = 5
    selected = []
    
    for _ in range(len(population)):
        # Choose indices of competitors from the population
        competitors_idx = np.random.choice(len(population), tournament_size, replace=False)
        competitors = population[competitors_idx]
        
        # Select the competitor with the best fitness
        winner = max(competitors, key=fitness)
        selected.append(winner)
        
    return np.array(selected)

# Populate the initial population
def populate(size):
    return np.random.randint(0, max_Troops + 1, (size, battlefields))

# Genetic Algorithm
def genetic_algorithm():
    population = populate(population_size)
    
    for gen in range(generations):
        population = selection(population)
        next_gen = []

        for i in range(0, population_size, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            child1, child2 = crossover(parent1, parent2)

            # Apply mutation and rebalance
            next_gen.append(rebalance(mutate(child1)))
            next_gen.append(rebalance(mutate(child2)))

        # Ensure all individuals are rebalanced before proceeding to the next generation
        population = np.array([rebalance(ind) for ind in next_gen])
        
    # Get the best allocation from the final generation
    best_allocation = max(population, key=fitness)
    return best_allocation

def calculate_score(currentHand, opponentHand):
    hand1 = 0
    hand2 = 0
    for idx in range(10):
        if currentHand[idx] > opponentHand[idx]:
            hand1 += idx + 1
        elif currentHand[idx] < opponentHand[idx]:
            hand2 += idx + 1
        else:
            hand1 += (idx + 1)/2
            hand2 += (idx + 1)/2
    return hand1

def generate_random_hand():
    hand = np.random.randint(0, 41, battlefields)
    return rebalance(hand)

def compare_against_random_opponents(best_allocation, num_opponents=100000):
    total_score = 0
    
    for _ in range(num_opponents):
        opponent_hand = generate_random_hand()  # Generate random opponent hand
        score = calculate_score(best_allocation, opponent_hand)  # Calculate score for this match
        total_score += score
    
    avg_score = total_score / num_opponents
    return avg_score
# Run the genetic algorithm
best_allocation = genetic_algorithm()

# Display results
print("Best Soldier Allocation: ", best_allocation)
print("Total Soldiers Allocated: ", sum(best_allocation))
print("Expected Points: ", fitness(best_allocation))

# Compare the best strategy against 500 random opponent hands
average_score = compare_against_random_opponents(best_allocation)

# Display results
print("Best Soldier Allocation: ", best_allocation)
print("Average Score Against Random Opponents: ", average_score)