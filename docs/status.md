---
layout: default
title: Status
---

### Summary of the Project
Our project is similar to that of an “escape room,” where the agent must find an exit of a room, where the room is provided as an input, within a reasonable amount of time. However, the agent must avoid death as there is a fire that spreads throughout the map and should aim to recover resources along the way. As a result, dying is equivalent to a penalty and recovering resources and finding exits within a short amount of time (or with minimal steps) is essential to rewards.  

As a status report, we decided to change the baseline of our progress such that our baseline is to have an agent that makes random moves. After accomplishing the baseline for this stage, the agent is able to find the exit in a minimal number of steps. Currently, we have not implemented any resources on the map.

### Approach

For this project, we decided to use the Deep-Q Learning algorithm which uses greedy-epsilon based policy. Deep-Q Learning is based on the following:

<p align="center">
    <img src="http://simplecore-dev.intel.com/ai/wp-content/uploads/sites/71/bellman-equation-example.png" alt="Bellman Ford Equation" />
</p>
<center><sub>Figure 1: Bellman Equation for Optimality<sub>1</sub></sub></center> 

We used this formula in our algorithm and defined the variables as followed:
* S: the current state which is the grid observation (ie a birds eye view of the map). It includes the information of our maze and the surroundings relative to the agent. Alternatively, the agent is able to see its surroundings as 9x9 grid, where the agent is at the center of it
* S’: Next state after taking action A’
* A: the current action that our agent will take. Currently, our agent can make four moves \[move north, move south, move east, move west\]. As of yet, it cannot pick up resources as it aims to find the exit in a minimal number of steps.
* A’: the future action 
* R: is our expected reward based on the action our agent takes.
* &gamma; (gamma): Discount factor. We set our gamma = 0.99 as that is the staple norm in Q-learning.

We used Tensorflow to build our Neural Network, and our network consists of three convolutional layers, followed by 2 fully connected layers. The last fully connected layer outputs values from 0-3 which is mapped to our action space. As for our input to the network, we used a little trick. Malmo offers a feature to observe the surrounding blocks by <ObservationFromGrid> tag in the mission XML file, and we get it from Malmo via a JSON file as a 2D array. We then convert that 2D array as a pixel representation of the grid and feed it to the Neural Network as a grayscale image. 

The techniques we used for our DQN is the Experience Replay Memory and uses two separate networks: Q-network and Target Network.  

* The Replay Memory consists of tuples of \[ “state”, “action”, “reward”, “next_state”, “done”\]. We then use the Replay Memory to sample a batch of size 32 to train our agent. Each step an agent takes is stored in the replay memory. In the beginning of training, we fill around 10% of the total replay memory with random exploration and gradually fills it as the agent trains. Once it is filled, the earliest memory is popped.

* The Target Network is used for calculating our loss and updates every N number of episodes. For our loss value, we use the squared differences of the Q-network and Target Network and performed gradient descent on it.
 
 <p align="center">
    <img src="http://simplecore-dev.intel.com/ai/wp-content/uploads/sites/71/q-learning-equation.png" alt="Loss Function" />
</p>
<center><sub>Figure 2: Loss Function<sub>2</sub></sub></center> 

To evaluate our progress, we used one of Tensorflow features, Tensorboard, to display the metrics. With the help of Tensorflow and Tensorboard in  our approach, we were able to obtain a graphical model of our data and its trends to ensure that our agent is working properly such that it fulfills our sanity case - that our agent does not constantly die in the fire.

Another big part of our project is generating a random maze. The user can choose their maze size and generate a random maze XML file. In order to avoid our agent from overfitting to only one map, we incorporated multiple random mazes into our training process. Every 20 episodes, we randomly choose from one of our randomly generated mazes and train the agent on that map. Even during Experience Replay Memory initialization, every 20 step, we change the maze. Doing this increases our training time and makes it harder for the agent to learn, but prevents the agent from overfitting to one specific map and help our agent to perform well on unseen test map. 


### Evaluation



### Remaining Goals and Challenges



### Resources Used

### Video

<iframe width="1280" height="720" src="https://www.youtube.com/embed/uiRR3c13AQ4" frameborder="0" allowfullscreen=""></iframe>


