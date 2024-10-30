

def hill_climbing():
    
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
    hand = np.random.randint(0, max_Troops + 1, battlefields)
    return rebalance(hand)

def compare_against_random_opponents(best_allocation, num_opponents=1000):
    total_score = 0
    
    for _ in range(num_opponents):
        opponent_hand = generate_random_hand()  # Generate random opponent hand
        score = calculate_score(best_allocation, opponent_hand)  # Calculate score for this match
        total_score += score
    
    avg_score = total_score / num_opponents
    return avg_score
