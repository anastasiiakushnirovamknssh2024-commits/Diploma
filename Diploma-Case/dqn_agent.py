import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Simple MLP for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(QNetwork, self).__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Stores past experiences for training
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# DQN agent with target network
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50000,
                 target_update=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.step_count = 0
        self.target_update = target_update

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state):
        # Epsilon-greedy policy
        self.step_count += 1
        eps_threshold = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-1.0 * self.step_count / self.epsilon_decay)
        if random.random() < eps_threshold:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# Training loop
def train_dqn(env, num_episodes=500, max_steps=1000, buffer_size=100000, batch_size=64):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=buffer_size)

    rewards_all = []
    lengths_all = []   # track episode lengths

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            agent.update(replay_buffer, batch_size=batch_size)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # force episode termination if done never triggered
        if not done:
            env.close()
            state = env.reset()

        rewards_all.append(total_reward)
        lengths_all.append(steps)   # store how many steps this episode lasted

        print(f"Episode {ep} — Reward: {total_reward:.2f}, Steps: {steps}")

    return rewards_all, lengths_all, agent
