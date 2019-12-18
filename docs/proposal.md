---
layout: default
title: Proposal
---
### Summary of the Project
  This project will be similar to that of an “escape room.”

*Input:*
   > * A (not necessarily randomly generated) map with blocks, spreading fire, and food resources. There will be multiple exits, and our agent should find an exit with a minimal number of steps, or within a fast time, without dying.
   > * The agent will know that it should avoid any death threats (equivalent to penalty).

*Output/Result:*
   > * A good result will have the agent escape from the fire in a minimal number of steps while having collected resources.
   > * A negative result would be the agent dying from the fire.

In terms of application(s),  we can apply this model to rescuing robots. A rescuing robot is a robot that can save human lives. Assuming that the environment can have single or multiple exits, our robot can find efficient escape routes, and simultaneously collect resources, while there is an eminent threat that grows in size. When robots learn how to save people and collect other valuable things in dangerous scenarios, we can avoid risking lives of people, such as firemen. In addition, there are many places that humans can not safely access, so using a robot to collect materials could be more efficient and safer.

### Algorithm 

We anticipate that we will be using reinforcement learning with a specificity in Q-learning. Since the agent will have little knowledge about its surroundings in the beginning, it will “learn” the series of best actions while playing the game multiple times.

### Evaluation

We will evaluate our agent’s performance based on how many steps it takes to find an exit and the total number of resources collected. 

**Quantitative Evaluation**

*Metrics and Baseline*

> Our quantitative metrics specifically aim at the number of steps taken to escape the map and the number of resources the agent  collects. The baseline is to have the agent safely escape the farm and to have it collect at least 1 resource. We decided to include a baseline for the resource; otherwise, the agent would simply escape the farm without collecting anything.

*Expectation in Performance Improvement*

> We expect to improve performance with fewer steps taken than the previous sessions and to collect more resources (~1-5 more) for performance of resource collecting. Ideally, we would like to have our agent improve in both metrics.

**Qualitative Analysis:**

*Sanity Cases*

> As sanity cases, we aim to check that the agent is not killing itself by running into the fires and is not running away from resources.

*Visualization of Internals of the Algorithm*

> To verify that our algorithm works, we will be recording the number of steps our agent takes in the farm and the amount of resources collected at each run. Both the number of steps and the number of resources collected will be a discrete integer. We will monitor the trend of both metrics while training.

*Moonshot Case:*

> We hope that our agent can perform well in more complicated worlds, not only in a situation where the environment is on fire.


### Appointment with the Instructor

22 April 2019, 10-10:15 am at DBH 4082



