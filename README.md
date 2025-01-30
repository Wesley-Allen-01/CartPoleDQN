# Cart Pole Balancing

In this project, I developed a Deep Q-Network to power an AI agent in an attempt to solve the cart pole problem which was introduced by Barto, Sutton, and Anderson in their 1983 paper *[Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems](https://ieeexplore.ieee.org/document/6313077)*.

![demo](https://gymnasium.farama.org/_images/cart_pole.gif)

The goal is to create an agent capable of balancing the pole in an upright position. At any point the agent can choose to either move left or right based off the cart's position, the cart's velocity, the pole's angle relative to the vertical axis, and the pole's angular velocity. 
An episode will terminate if at any point, the cart's x-coordinate leaves the (-2.4, 2.4) range or the pole angle exceeds the (-12, 12) degree range. 

## Deep Q-Networks
Deep Q-Networks or DQNs bridge the gap between traditional Q-learning and deep neural networks. These networks take, as input, environment data, and output a "quality score" of each action the agent could take at a particular state. As the agent gains more experience, the outputted quality scores become more accurate enabling greater agent performance in a particular environment. 
