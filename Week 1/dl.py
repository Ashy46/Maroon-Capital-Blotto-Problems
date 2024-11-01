import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class BlottoEnv:
    def __init__(self, num_battlefields=10, max_troops=100):
        self.num_battlefields = num_battlefields
        self.max_troops = max_troops

    def reset(self):
        # Reset environment for a new episode
        self.state = np.zeros(self.num_battlefields, dtype=np.int32)
        return self.state

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

    def step(self, allocation):
        # Generate a random opponent allocation
        opponent_allocation = np.random.randint(0, 41, size=self.num_battlefields)
        opponent_allocation = rebalance(opponent_allocation)
        # Calculate the reward (player's score) using the new scoring logic
        reward = self.calculate_reward(allocation, opponent_allocation)
        return reward, opponent_allocation

    def calculate_reward(self, currentHand, opponentHand):
        hand1 = 0  # player score
        hand2 = 0  # opponent score

        # Calculate scores based on allocation comparison
        for idx in range(self.num_battlefields):
            if currentHand[idx] > opponentHand[idx]:
                hand1 += idx + 1
            elif currentHand[idx] < opponentHand[idx]:
                hand2 += idx + 1
            else:
                hand1 += (idx + 1) / 2
                hand2 += (idx + 1) / 2

        # Return the player's score as the reward
        return hand1

class PolicyNetwork(nn.Module):
    def __init__(self, num_battlefields):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_battlefields, 128)
        self.fc2 = nn.Linear(128, num_battlefields)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1) * 100  # Ensures outputs sum close to max_troops
        return x

def train(env, policy_net, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        state = torch.FloatTensor(env.reset())
        optimizer.zero_grad()
        
        # Get allocation distribution (action probabilities)
        allocations = policy_net(state)
        allocations = allocations / allocations.sum() * env.max_troops  # Normalize to max_troops

        # Get reward
        reward, opponent_allocation = env.step(allocations.detach().numpy())
        
        # Calculate the loss (negative reward times log-probability)
        loss = -torch.sum(torch.log(allocations) * reward)
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f'Episode {episode}: Reward = {reward}, Opponent Allocation = {opponent_allocation}')

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
    for _ in range(num_games):
        state = torch.FloatTensor(env.reset())
        allocations = policy_net(state).detach().numpy()
        allocations = allocations / allocations.sum() * env.max_troops
        reward, _ = env.step(allocations)
        total_reward += reward

    avg_reward = total_reward / num_games
    print(f'Average Reward after evaluation: {avg_reward}')

evaluate(env, policy_net)

