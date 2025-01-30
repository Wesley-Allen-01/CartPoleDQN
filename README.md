# Cart Pole Balancing

In this project, I developed a Deep Q-Network to power an AI agent in an attempt to solve the cart pole problem which was introduced by Barto, Sutton, and Anderson in their 1983 paper *[Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems](https://ieeexplore.ieee.org/document/6313077)*.

![demo](https://gymnasium.farama.org/_images/cart_pole.gif)

The goal is to create an agent capable of balancing the pole in an upright position. At any point the agent can choose to either move left or right based off the cart's position, the cart's velocity, the pole's angle relative to the vertical axis, and the pole's angular velocity. 
An episode will terminate if at any point, the cart's x-coordinate leaves the (-2.4, 2.4) range or the pole angle exceeds the (-12, 12) degree range. 

## Deep Q-Networks
Deep Q-Networks or DQNs bridge the gap between traditional Q-learning and deep neural networks. These networks take, as input, environment data, and output a "quality score" of each action the agent could take at a particular state. As the agent gains more experience, the outputted quality scores become more accurate enabling greater agent performance in a particular environment. The agent learns by comparing the predicted Q-values from a policy network with Q-values calculated using the Bellman Equation which takes as input Q-values from a target network. The Bellman equation is shown below.

$$Q(s, a) = r + \gamma \cdot \max(Q(s', a'))$$

The variable $r$ indicates the immediate reward an agent receives by taking action $a$ at state $s$. The variable $\gamma$ represents a discount factor, which we use to balance the model's preference for immediate rewards and future rewards. This discount factor is multiplied by $\max(Q(s', a'))$ which is the maximum Q-value out of any action in the state that we would arrive in after taking action $a$ at state $s$.

As mentioned previously, I utilized two separate networks: a policy network and a target network. During training, the parameters of the policy network are updated at each time step. To do this, we sample 128 scenarios from the Replay Memory, and predict Q-values for each. The replay memory stores the next state we arrived in after the states in the batch, so we use those to predict the next states Q-values using the target network. The next state's Q-values are inputted into the Bellman equation which enables us to determine what the Q-values at the preceeding state should be. Our loss function, which in our case was the `nn.SmoothL1Loss`, computes the difference between the policy net's predicted Q-values and the Q-values derived from the Bellmman equation.
