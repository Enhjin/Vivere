---
layout: default
title: Proposal
---
## Summary of the project
  This project will create something similar to that of an “escape room.”

  * Input:
       * A randomly generated farm that is on fire. Our agent should find the exit in the shortest time and try its best to save resources while running away, such as collecting eggs or plants while in danger.
       * The agent will know it should avoid any death threats (equivalent to penalty) and should collect resources (equivalent to rewards)
       * As of current, we decided to have only one exit, but this may change as the project progresses.

  * Output/Result:
       * A good result will have the agent escape from the fire and would have the agent collect as many resources as possible.
       * A negative result would be the agent dying in the farm.

In terms of application(s),  we can apply this model to rescuing robots. A rescuing robot is a robot that can save human lives. When robots learn how to save people and collect other valuable things in dangerous scenarios, we can avoid risking lives of firemen. In addition, there are many places that humans can not safely access, so using a robot to collect materials for us could be more efficient and safer.

## Algorithm 

We anticipate on using reinforcement learning with a specificity in Q-learning. Since the agent have little knowledge about its surroundings in the beginning, it will “learn” the series of best actions while playing the game multiple times.

## Evaluation

We will evaluate our agent’s performance based on how much time is spent on each farm and the total number of resources collected.

**Quantitative Evaluation**

*Metrics and Baseline*

Our quantitative metrics specifically aim at the time it takes to escape the farm and the number of resources the agent  collects. The baseline is to have the agent safely escape the farm and to have it collect at least 1 resource. We decided to include a baseline for the resource - at least collect 1 resource - otherwise, the agent would simply escape the farm without collecting anything.

*Expectation in Performance Improvement*

 We expect to improve performance of time time by a couple seconds faster (~2-5 seconds faster) and to collect more resources (~1-5 more) for performance of resource collecting. Ideally, we would like to have our agent improve in both metrics.

**Qualitative Analysis:**

*Sanity Cases*

As sanity cases, we aim to check that the agent is not killing itself by running into the fires and is not running away from resources.

*Visualization of Internals of the Algorithm*

To verify that our algorithm works, we will be recording the time the agent spends in the farm and the amount of resources collected at each run. The time will be continuous while the number of resources collected will be a discrete integer. We will monitor the trend of both metrics while training.

*Moonshot Case:*

We hope that our agent can perform well in more complicated worlds, not only in an environment where a farm is on fire.


## Appointment with the Instructor

22 April 2019, 10-10:15 am at DBH 4082
