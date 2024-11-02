import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BlottoEnv:
    def __init__(self, num_battlefields=10, max_troops=100):
        self.num_battlefields = num_battlefields
        self.max_troops = max_troops

    def reset(self):
        # Reset environment for a new episode
        self.state = np.zeros(self.num_battlefields, dtype=np.int32)
        return self.state

    def rebalance(self, setUp):
        total_troops = sum(setUp)

        # If troops exceed max_Troops, remove troops randomly from positive indices
        while total_troops > self.max_troops:
            positive_indices = np.where(np.array(setUp) > 0)[0]  # Find indices with more than 0 troops
            if len(positive_indices) > 0:
                remove_from = np.random.choice(positive_indices)
                setUp[remove_from] -= 1
                total_troops -= 1
            else:
                break

        # If troops are less than max_Troops, add troops randomly to any index
        while total_troops < self.max_troops:
            add_to = np.random.randint(len(setUp))
            setUp[add_to] += 1
            total_troops += 1

        return setUp

    def step(self, allocation):
        # Generate a random opponent allocation
        opponent_allocation = [4, 5, 8, 10, 12, 1, 24, 34, 1, 1]
        opponent_allocation = self.rebalance(opponent_allocation)
        reward = self.calculate_reward(allocation, opponent_allocation)
        return reward, opponent_allocation

    def calculate_reward(self, currentHand, opponentHand):
        hand1 = 0  # player score
        hand2 = 0  # opponent score

        for idx in range(self.num_battlefields):
            if currentHand[idx] > opponentHand[idx]:
                hand1 += idx + 1
            elif currentHand[idx] < opponentHand[idx]:
                hand2 += idx + 1
            else:
                hand1 += (idx + 1) / 2
                hand2 += (idx + 1) / 2

        return hand1

class PolicyNetwork(nn.Module):
    def __init__(self, num_battlefields):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_battlefields, 128)
        self.fc2 = nn.Linear(128, num_battlefields)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1) * 100  # Ensures outputs are scaled
        return x

def integer_allocation(allocation, max_troops):
    # Round allocations and ensure they sum to max_troops
    rounded_allocation = np.round(allocation).astype(int)
    rounded_allocation = env.rebalance(rounded_allocation)
    return rounded_allocation

def allocate_troops(raw_allocations, max_troops):
    # Apply a softmax to get probabilities and scale to max_troops
    allocations = np.exp(raw_allocations)  # Using exp to convert logits to probabilities
    allocations /= np.sum(allocations)      # Normalize
    allocations *= max_troops               # Scale to max_troops
    
    # Round to integers and ensure sum is max_troops
    allocations = np.round(allocations).astype(int)

    # Rebalance to ensure it meets the maximum troop requirement
    total_troops = allocations.sum()
    if total_troops > max_troops:
        while total_troops > max_troops:
            # Reduce allocation randomly
            index = np.random.choice(np.where(allocations > 0)[0])
            allocations[index] -= 1
            total_troops -= 1
    elif total_troops < max_troops:
        while total_troops < max_troops:
            # Increase allocation randomly
            index = np.random.choice(len(allocations))
            allocations[index] += 1
            total_troops += 1

    return allocations


def train(env, policy_net, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        best_allocation = None
        best_reward = float('-inf')
        state = torch.FloatTensor(env.reset())
        optimizer.zero_grad()

        # Get raw allocation distribution (logits)
        raw_allocations = policy_net(state)

        # Convert raw allocations to troop allocations
        allocations = allocate_troops(raw_allocations.detach().numpy(), env.max_troops)

        # Convert allocations to a tensor for reward calculation
        allocations_tensor = torch.FloatTensor(allocations)

        # Get reward
        reward, opponent_allocation = env.step(allocations)

        # Calculate the loss (negative reward)
        # Use raw allocations for calculating the loss
        loss = -torch.sum(torch.log(raw_allocations + 1e-10) * reward)  # Avoid log(0)
        loss.backward()
        optimizer.step()

        if reward > best_reward:
            best_reward = reward
            best_allocation = allocations

        if episode % 100 == 0:
            print(f'Episode {episode}: Reward = {reward}, Opponent Allocation = {opponent_allocation}')
    
    print("Best Allocation:", best_allocation)
    print("Training complete.")


# Initialize environment, policy, and optimizer
num_battlefields = 10
max_troops = 100
env = BlottoEnv(num_battlefields, max_troops)
policy_net = PolicyNetwork(num_battlefields)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# Train the model
train(env, policy_net, optimizer)

def evaluate(env, policy_net, num_games=500):
    total_reward = 0
    best_allocation = None
    best_reward = float('-inf')

    for _ in range(num_games):
        state = torch.FloatTensor(env.reset())
        allocations = policy_net(state).detach().numpy()
        integer_allocations = integer_allocation(allocations, env.max_troops)
        reward, _ = env.step(integer_allocations)

        total_reward += reward
        
        # Track the best allocation during evaluation
        if reward > best_reward:
            best_reward = reward
            best_allocation = integer_allocations

    avg_reward = total_reward / num_games
    print(f'Average Reward after evaluation: {avg_reward}')
    print("Best Allocation during evaluation:", best_allocation)
    print("Best Reward during evaluation:", best_reward)

evaluate(env, policy_net)
