from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #1: Run simple mission

from builtins import range
import MalmoPython
import os
import sys
import time

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
              <ModSettings>
                  <MsPerTick>75</MsPerTick>
              </ModSettings>
              
              <ServerSection>
                <ServerInitialConditions>
                  <Time>
                    <StartTime>16000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                  </Time>
                  <Weather>clear</Weather>
                  <AllowSpawning>true</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                  <DrawingDecorator>
                    <DrawCuboid x1="-11" y1="226" z1="-6" x2="0" y2="246" z2="5" type="netherrack"/>
                    <DrawCuboid x1="-10" y1="227" z1="-5" x2="-1" y2="246" z2="4" type="air"/>
                    <DrawCuboid x1="-10" y1="227" z1="-5" x2="-1" y2="227" z2="4" type="carpet"/>

                    <DrawCuboid x1="-6" y1="227" z1="4" x2="-6" y2="229" z2="4" type="emerald_block"/>
                    <DrawCuboid x1="-1" y1="227" z1="2" x2="-1" y2="229" z2="2" type="emerald_block"/>
                    <DrawCuboid x1="-10" y1="227" z1="1" x2="-10" y2="229" z2="1" type="emerald_block"/>
                    <!-- First Row-->
                    <DrawCuboid x1="-4" y1="227" z1="4" x2="-4" y2="227" z2="4" type="sea_lantern"/>
                    <!-- Second Row -->
                    <DrawCuboid x1="-5" y1="227" z1="3" x2="-5" y2="227" z2="3" type="sea_lantern"/>
                    <DrawCuboid x1="-2" y1="227" z1="3" x2="-2" y2="227" z2="3" type="sea_lantern"/>
                    <DrawCuboid x1="-7" y1="227" z1="3" x2="-7" y2="227" z2="3" type="sea_lantern"/>
                    <DrawCuboid x1="-9" y1="227" z1="3" x2="-9" y2="227" z2="3" type="sea_lantern"/>
                    <DrawCuboid x1="-10" y1="227" z1="3" x2="-10" y2="227" z2="3" type="sea_lantern"/>
                    <!-- Third Row -->
                    <DrawCuboid x1="-2" y1="227" z1="2" x2="-2" y2="227" z2="2" type="sea_lantern"/>
                    <DrawCuboid x1="-4" y1="227" z1="2" x2="-4" y2="227" z2="2" type="sea_lantern"/>
                    <DrawCuboid x1="-8" y1="227" z1="2" x2="-8" y2="227" z2="2" type="sea_lantern"/>
                    <!-- Fourth Row -->
                    <DrawCuboid x1="-1" y1="227" z1="1" x2="-1" y2="227" z2="1" type="sea_lantern"/>
                    <DrawCuboid x1="-4" y1="227" z1="1" x2="-4" y2="227" z2="1" type="sea_lantern"/>
                    <DrawCuboid x1="-6" y1="227" z1="1" x2="-6" y2="227" z2="1" type="sea_lantern"/>
                    <!-- Fifth Row -->
                    <DrawCuboid x1="-2" y1="227" z1="0" x2="-2" y2="227" z2="0" type="sea_lantern"/>
                    <DrawCuboid x1="-6" y1="227" z1="0" x2="-6" y2="227" z2="0" type="sea_lantern"/>
                    <DrawCuboid x1="-7" y1="227" z1="0" x2="-7" y2="227" z2="0" type="sea_lantern"/>
                    <DrawCuboid x1="-10" y1="227" z1="0" x2="-10" y2="227" z2="0" type="sea_lantern"/>
                    <!-- Sixth Row -->
                    <DrawCuboid x1="-1" y1="227" z1="-1" x2="-1" y2="227" z2="-1" type="sea_lantern"/>
                    <DrawCuboid x1="-3" y1="227" z1="-1" x2="-3" y2="227" z2="-1" type="sea_lantern"/>
                    <DrawCuboid x1="-5" y1="227" z1="-1" x2="-5" y2="227" z2="-1" type="sea_lantern"/>
                    <DrawCuboid x1="-7" y1="227" z1="-1" x2="-7" y2="227" z2="-1" type="sea_lantern"/>
                    <DrawCuboid x1="-8" y1="227" z1="-1" x2="-8" y2="227" z2="-1" type="sea_lantern"/>
                    <DrawCuboid x1="-10" y1="227" z1="-1" x2="-10" y2="227" z2="-1" type="sea_lantern"/>
                    <!-- Seventh Row -->
                    <DrawCuboid x1="-2" y1="227" z1="-2" x2="-2" y2="227" z2="-2" type="sea_lantern"/>
                    <DrawCuboid x1="-5" y1="227" z1="-2" x2="-5" y2="227" z2="-2" type="sea_lantern"/>
                    <DrawCuboid x1="-10" y1="227" z1="-2" x2="-10" y2="227" z2="-2" type="sea_lantern"/>
                    <!-- Eigth Row -->
                    <DrawCuboid x1="-4" y1="227" z1="-3" x2="-4" y2="227" z2="-3" type="sea_lantern"/>
                    <DrawCuboid x1="-5" y1="227" z1="-3" x2="-5" y2="227" z2="-3" type="sea_lantern"/>
                    <DrawCuboid x1="-7" y1="227" z1="-3" x2="-7" y2="227" z2="-3" type="sea_lantern"/>
                    <DrawCuboid x1="-9" y1="227" z1="-3" x2="-9" y2="227" z2="-3" type="sea_lantern"/>
                    <DrawCuboid x1="-10" y1="227" z1="-3" x2="-10" y2="227" z2="-3" type="sea_lantern"/>
                    <!-- Ninth Row -->
                    <DrawCuboid x1="-2" y1="227" z1="-4" x2="-2" y2="227" z2="-4" type="sea_lantern"/>
                    <DrawCuboid x1="-7" y1="227" z1="-4" x2="-7" y2="227" z2="-4" type="sea_lantern"/>
                    <DrawCuboid x1="-9" y1="227" z1="-4" x2="-9" y2="227" z2="-4" type="sea_lantern"/>
                    <!-- Tenth Row -->
                    <DrawCuboid x1="-6" y1="227" z1="-5" x2="-6" y2="227" z2="-5" type="sea_lantern"/>

                    <DrawCuboid x1="0" y1="227" z1="5" x2="-11" y2="246" z2="5" type="beacon"/>
                    <DrawCuboid x1="0" y1="227" z1="-6" x2="-11" y2="246" z2="-6" type="beacon"/>
                    <DrawCuboid x1="0" y1="227" z1="4" x2="0" y2="246" z2="-5" type="beacon"/>
                    <DrawCuboid x1="-11" y1="227" z1="4" x2="-11" y2="246" z2="-5" type="beacon"/>

                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                  <ServerQuitFromTimeUp timeLimitMs="1000"/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                  <Name>Survivor</Name>
                  <AgentStart>
                      <Placement x="-4" y="227" z="-5" yaw="0"/>
                      <Inventory>
                      </Inventory>
                  </AgentStart>
                  <AgentHandlers>
                      <ObservationFromFullStats/>
                      <DiscreteMovementCommands/>
                      <ObservationFromNearbyEntities>
                          <Range name="zombies" xrange="10" yrange="2" zrange="10" />
                      </ObservationFromNearbyEntities>
                      <RewardForMissionEnd rewardForDeath="-1000">
                          <Reward description="out_of_time" reward="0"/>
                      </RewardForMissionEnd>
                      <MissionQuitCommands quitDescription="give_up"/>
                  </AgentHandlers>
              </AgentSection>
              
            </Mission>'''

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()
my_mission.drawBlock( -2,227,-5,"fire")
# my_mission.drawBlock( -92,227,-1,"fire")
# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
# agent_host.sendCommand("turn -0.5")
# agent_host.sendCommand("move 1")

while not world_state.has_mission_begun:
    # agent_host.sendCommand("pitch 1")
    time.sleep(1)
    # agent_host.sendCommand("attack 1")
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

# Loop until mission ends:
while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission ended")
# Mission has ended.
