---
layout: default
title:  Home
---

<p align="center">
    <img src="https://gonintendo.com/system/file_uploads/uploads/000/041/527/original/minecraft-600x333.png" alt="Minecraft" />
</p>


### Welcome to Vivere's Team Page! 

#### Our team consists of three members:

> * Enkhjin Enkhjargal
> * Maxine Yuan
> * Xinxin Xie

---

### What is Vivere?
Vivere in Latin means "life." We chose this based on our project idea: an artificial intelligence, built within malmo, will be trapped in a maze and must escape. However, within this maze, is an increasingly spreading fire and resources. Our aim for our AI is to be able to find an exist (out of multiple ones) and escape in a fast time while having collected food resources.

For more information and details, visit the [Proposal](proposal.html) page.

---

### Beginning & Status
To start this project, we first created random maps by writing a program that helps us generate a map via XML strings. The user would input the size of the map he or she wants, and shortly after, the program will output a long string for the map. This map includes exit point(s), blocks/areas where the agent is not able to access, and a location for the fire to start spreading.  To ensure our agent is not overfitting on one kind of map, we used this program to generate random maps for our agent to train and test. After the maze is built, we spawn the agent in a random location. The image below shows the final result of a randomly generated 5x5 sized maze. 

<p align="center">
    <img src="https://raw.githubusercontent.com/Enhjin/Vivere/master/homepage%20img.png" alt="Result" />
</p>

To see more of our status report, visit the [Status](status.html) page.


Source code: [Github repo](https://github.com/Enhjin/Vivere "Github repo").
