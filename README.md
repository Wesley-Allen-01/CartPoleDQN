# Cart Pole Balancing

In this project, I developed a Deep Q-Network to power an AI agent in an attempt to solve the cart pole problem which was introduced by Barto, Sutton, and Anderson in their 1983 paper *[Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems](https://ieeexplore.ieee.org/document/6313077)*.

![demo](https://gymnasium.farama.org/_images/cart_pole.gif)

The goal is to create an agent capable of balancing the pole in an upright position. At any point the agent can choose to either move left or right based off the cart's position, the cart's velocity, the pole's angle relative to the vertical axis, and the pole's angular velocity. 
An episode will terminate if at any point, the cart's x-coordinate leaves the (-2.4, 2.4) range or the pole angle exceeds the (-12, 12) degree range. 

## Deep Q-Networks
Deep Q-Networks or DQNs bridge the gap between traditional Q-learning and deep neural networks. These networks take, as input, environment data, and output a "quality score" of each action the agent could take at a particular state. As the agent gains more experience, the outputted quality scores become more accurate enabling greater agent performance in a particular environment. The agent learns by comparing the predicted Q-values from a policy network with Q-values calculated using the Bellman Equation which takes as input Q-values from a target network. The Bellman equation is shown below.

$$Q(s, a) = r + \gamma \cdot \max(Q(s', a'))$$

The variable $r$ indicates the immediate reward an agent receives by taking action $a$ at state $s$. The variable $\gamma$ represents a discount factor, which we use to balance the model's preference for immediate rewards and future rewards. This discount factor is multiplied by $\max(Q(s', a'))$ which is the maximum Q-value out of any action in the state that we would arrive in after taking action $a$ at state $s$.

As mentioned previously, I utilized two separate networks: a policy network and a target network. During training, the parameters of the policy network are updated at each time step. To do this, we sample 128 scenarios from the Replay Memory, and predict Q-values for each. The replay memory stores the next state we arrived in after the states in the batch, so we use those to predict the next states Q-values using the target network. The next state's Q-values are inputted into the Bellman equation which enables us to determine what the Q-values at the preceeding state should be. Our loss function, which in our case was the `nn.SmoothL1Loss`, computes the difference between the policy net's predicted Q-values and the Q-values derived from the Bellmman equation. We use the Adam optimizer, with amsgrad set to True, to optimize the parameters of the model to minimize loss. After training for 350 timesteps, the agent was able to regularly achieve a max score of 500, representing balancing the pole for 10 seconds.

## Soft updates vs Hard Updates
My initial implementation utilized a "hard" updating strategy for the target net. This mean that at every $t$ time steps, I would completely copy all of the parameters of the policy network onto the target network. I believe that this sudden shift destabalized the gradients which greatly lessened the agent's ability to learn. After a few hours of punching the air, I decided to switch to a soft-updating strategy. In this strategy, we gradually blend the parameters of the policy network into the target network, minimizing sudden shifts, and stabilizing gradients. This can be explained mathematically like so

$$\theta_{target} \leftarrow \tau \cdot \theta_{policy} + (1-\tau) \cdot \theta_{policy}$$

This switch greatly improved the agent's learning ability resulting in an agent capable of scoring between 150-300.

## Custom Reward Function
By default, the agent receives a reward of +1 for each timestep that the cart stays balanced and within the x-coordinate range. My first successful training session resulted in a model that was able to keep the pole balanced for 150-300 time steps. However, I observed that, in nearly every case, the agent's session was terminated because it moved too far to the right. I preserved this model, and it is stored in the models folder under the name swing_to_the_right.pth. To work around this, I decided to implement a custom reward function for the agent that differed from the default function offered by the gymnasium library. In my reward function, I lessened the reward received by the agent when it moved far from the origin and when the pole moved off the vertical axis. This function is in cart_pole.ipynb. This greatly improved the model resulting in a much more stable agent which regularly maxed out the score in this environment. 

## References

- Barto, Sutton, and Anderson (1983). *[Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems](https://ieeexplore.ieee.org/document/6313077)*
- Mnih, Kavukcuoglu, Dilver, Graves, Antonoglou, Wierstra, Riedmiller (2013) *[Playing Atari with Deep Reinforcing Learning](https://arxiv.org/abs/1312.5602)*
- [Cart Pole Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
