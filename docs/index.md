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
Vivere in Latin means "life." We chose this based on our project idea: an artificial intelligence, built within malmo, will be trapped in a maze and must escape. However, within this maze, is a spreading fire. Our aim for our AI is to be able to find an exist (out of multiple ones) and escape in a fast time.

For more information and details, visit the [Proposal](proposal.html) page.

---

### Beginning & Status
To start this project, we first created random maps by writing a program that helps us generate a map via XML strings. The user inputs the size of the map he or she wants, and shortly after, the program will output a long string for the map. This map includes exit point(s), blocks/areas where the agent is unable to access, and a location for the fire to start spreading. It is important to note the material of the generated map. We decided to set the material of the floor to be carpet so that the fire can spread; however, in Minecraft, it is normal for the fire to go out after a certain period of time. To battle this, we put netherrack blocks below the carpet so that when the carpet burns out, the fire continues and never estinguishes. In order to have our fire spread in a somewhat 'controlled path' so that it does not spread everywhere, we used sea lanterns to prevent the fire from haphazardly spreading in any direction. The image below shows the final result of a randomly generated 5x5 sized maze.

<p align="center">
    <img src="https://raw.githubusercontent.com/Enhjin/Vivere/master/homepage%20img.png" alt="Result"  width="600" height="600"/>
</p>

To see more of our status report, visit the [Status](status.html) page.


### Final Report
Our final result of our agent is able to solve 7x7 mazes effectively. We readjusted our random map generator to accomodate variables that we cannot control (eg the direction and speed of how fast the fire spreads). By doing so, we were able to successfully train our agent on 7x7 maps, and with proper adjustments, our training did not result in an underfitting and/or overfitting model.

To read our complete process and project, visit the [Final](final.html) page.

Alternatively, we also included a summarized video report of our final product.

<iframe width="1280" height="720" src="https://www.youtube.com/embed/5i4LMlA8y80" frameborder="0" allowfullscreen=""></iframe>



Source code: [Github repo](https://github.com/Enhjin/Vivere "Github repo").
