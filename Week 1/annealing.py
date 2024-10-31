#imports
import numpy as np
import random

solution = [4, 5, 8, 10, 12, 1, 24, 34, 1, 1]
best_allocation = solution
def_temperature = 90
COOLING_RATE = 0.98
max_iterations = 1000


def simulated_annealing(init_solution, max_iter, temperature):
    best_solution = [4, 5, 8, 10, 12, 1, 24, 34, 1, 1]
    for i in range(max_iter):
        print(i)
        current_solution = init_solution
        new_solution = mutate(current_solution)

        current_energy = fitness(current_solution)
        new_energy = fitness(new_solution)

        acceptance_probability = calculate_Probability(current_energy, new_energy, temperature)
        if acceptance_probability > random.uniform(0, 1):
            current_solution = new_solution

        temperature *= COOLING_RATE

        if fitness(current_solution) > fitness(best_solution):
            best_solution = current_solution
    
    return best_solution


def calculate_Probability(current_energy, new_energy, temperature):
    if new_energy > current_energy:
        return 1.0
    else:
        return np.exp((new_energy - current_energy) / temperature)

#Instead of using selection allocations, I will just randomly compare against 1000 opponents
def fitness(setUp):
    avg = compare_against_random_opponents(setUp)
    return avg

#similar to mutate from genetic but guarenteed small changes 
def mutate(setUp):
    mutation_index = np.random.randint(10)
        
    # Calculate the maximum possible troop number that can be allocated to the mutation index
    max_allocation = np.random.randint(0, 7)

    setUp[mutation_index] += max_allocation    
        
    return rebalance(setUp)

def rebalance(setUp):
    total_troops = sum(setUp)

    # If troops exceed max_Troops, remove troops randomly from positive indices
    while total_troops > 100:
        positive_indices = np.where(np.array(setUp) > 0)[0]  # Find indices with more than 0 troops
        if len(positive_indices) > 0:
            remove_from = np.random.choice(positive_indices)
            setUp[remove_from] -= 1
            total_troops -= 1

    # If troops are less than max_Troops, add troops randomly to any index
    while total_troops < 100:
        add_to = np.random.randint(len(setUp))  # Randomly select any index to add a troop
        setUp[add_to] += 1
        total_troops += 1

    return setUp

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
    hand = np.random.randint(0, 101, size=10)
    return rebalance(hand)

def compare_against_random_opponents(best_allocation, num_opponents=1000):
    total_score = 0
    
    for _ in range(num_opponents):
        opponent_hand = generate_random_hand()  # Generate random opponent hand
        score = calculate_score(best_allocation, opponent_hand)  # Calculate score for this match
        total_score += score
    
    avg_score = total_score / num_opponents
    return avg_score

best_allocation = simulated_annealing(solution, max_iterations, def_temperature)
print("Best Solution: ", best_allocation)
print("Total Soldiers: ", sum(best_allocation))
print("Avg Against Random Opponents: ", compare_against_random_opponents(best_allocation, 100000))