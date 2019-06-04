#Created by Xinxin Xie on 05/24/2019

#!/usr/bin/env python
from builtins import range
import MalmoPython
import os
import sys
import time
import malmoutils
import generate_maze as gm

all_xml = []
for i in range(40): #rotation makes 3 more maps
    maze = gm.Maze(7, 7)
    xml = maze.generate_all_xml()
    all_xml.extend(xml)

n = len(all_xml)
for i in range(n):
    fn = "maze{0}.xml".format(i)
    f= open(fn,"w")
    XML = '''
    <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                  <About>
                    <Summary>Survive the farm fire!</Summary>
                  </About>

                  <ModSettings>
                      <MsPerTick>1</MsPerTick>
                  </ModSettings>

                  <ServerSection>
                    <ServerInitialConditions>
                      <Time>
                        <StartTime>6000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                      </Time>
                      <Weather>clear</Weather>
                      <AllowSpawning>true</AllowSpawning>
                    </ServerInitialConditions>
                    <ServerHandlers>

                      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                      <DrawingDecorator>
                          {0}

                      </DrawingDecorator>
                      <ServerQuitWhenAnyAgentFinishes/>
                      <ServerQuitFromTimeUp timeLimitMs="900000"/>
                    </ServerHandlers>
                  </ServerSection>

                  <AgentSection mode="Survival">
                      <Name>Survivor</Name>
                      <AgentStart>
                          <Placement x="3.5" y="227" z="1.5" yaw="0"/>
                          <Inventory>
                          </Inventory>
                      </AgentStart>
                      <AgentHandlers>
                          <ObservationFromFullStats/>
                          <DiscreteMovementCommands>
                              <ModifierList type="deny-list">
                                <command>attack</command>
                              </ModifierList>
                          </DiscreteMovementCommands>
                          <VideoProducer want_depth="false">
                              <Width>640</Width>
                              <Height>480</Height>
                          </VideoProducer>
                          <ObservationFromGrid>
                              <Grid name="floor10x10">
                                  <min x="-6" y="0" z="-6"/>
                                  <max x="6" y="0" z="6"/>
                              </Grid>
                          </ObservationFromGrid>
                          <RewardForTouchingBlockType>
                              <Block reward="100.0" type="emerald_block"/>
                              <Block reward="-50.0" type="fire"/>
                              <Block reward="-2.0" type="sea_lantern"/>
                              <Block reward="-2.0" type="netherrack"/>
                              <!-- <Block reward="100.0" type="carpet"/> -->
                          </RewardForTouchingBlockType>
                          <RewardForSendingCommand reward="-1"/>
                          <RewardForMissionEnd rewardForDeath="-10.0">
                              <Reward description="found_goal" reward="100.0" />
                              <Reward description="out_of_time" reward="-5.0" />
                          </RewardForMissionEnd>
                          <AgentQuitFromTouchingBlockType>
                              <Block type="emerald_block" description="found_goal" />
                          </AgentQuitFromTouchingBlockType>
                      </AgentHandlers>
                  </AgentSection>
                </Mission> '''.format(all_xml[i])
    f.write(XML)


