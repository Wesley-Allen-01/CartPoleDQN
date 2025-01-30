from model import DQN
import torch
import torch.nn as nn
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
policy_net = DQN(n_observations, n_actions)

policy_net.load_state_dict(torch.load('models/model.pth'))
policy_net.eval()

state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
done = False


testing_episodes = 10

for i in range(testing_episodes):
    total_reward = 0
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    while not done:
        with torch.no_grad():
            action = policy_net(state).argmax().item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    print(f'Total reward: {total_reward}')

env.close()