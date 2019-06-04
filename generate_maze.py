#Created by Xinxin Xie on 05/14/2019

import random
import argparse
random.seed()

class Maze:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.key_grid = []
        self.grid = []
        self.grids = []

    def init_str(self):
        init_xml_1 = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="246" z2="{3}" type="glass"/>\n'.format(0, self.w+1, 0, self.h+1)
        init_xml_2 = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="246" z2="{3}" type="air"/>\n'.format(1, self.w, 1, self.h)
        init_xml_3 = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="sea_lantern"/>\n'.format(1, self.w, 1, self.h)
        return init_xml_1 + init_xml_2 + init_xml_3

    def get_grid(self):
        if len(self.grid) == 0:
            self.create_maze()
        return self.grid

    def get_key_cells(self):
        if len(self.key_grid) == 0:
            self.create_maze()
        return self.key_grid


    def generate_random_grids(self): #randomized_prim algorithms from wikipedia
        #1. start with a grid full of walls
        grid = [[0] * self.w for _ in range(self.h)]  # generate records, 0 means a block, 1 means a carpet
        # grids = [[0] * (w + 2)] + grids + [[0] * (w + 2)]  # surrounding grids with 0's

        assert self.h >= 5
        assert self.w >= 5
        #2. pick a cell,mark it as part of the passage.
        start_x = random.randrange(2, self.h-2)
        start_y = random.randrange(2, self.w-2)

        grid[start_x][start_y] = 1 # 1 means carpet, a part of the maze
        frontier_cells = [(start_x - 2, start_y), (start_x, start_y + 2), (start_x + 2, start_y), (start_x, start_y - 2)]
        for (i, j) in frontier_cells[:]:
            if (i > (self.h - 1) or i < 0 or j > (self.w - 1) or j < 0) or grid[i][j] == 1:  #within the grid, cannot be carpet
                frontier_cells.remove((i, j))

        while(len(frontier_cells) != 0):#While there are walls in the list:

            (x, y) = random.choice(frontier_cells) # Pick a random wall from the list.
            #get neighbor with a distance of two of the frontier cell
            neighbors_cells = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)] # in the beginning, there must have one being 1

            for (i, j) in neighbors_cells[:]:
                if (i > (self.h - 1) or i < 0 or j > (self.w - 1) or j < 0) or grid[i][j] == 0:
                    neighbors_cells.remove((i, j))

            (a, b) = random.choice(neighbors_cells) #choose one neighbor
            (c, d) = ((x+a)//2, (y+b)//2)

            grid[x][y] = 1 #frontier cell becomes a part of path
            grid[c][d] = 1 #cell in the middle is a part of path
            new_frontier_cells = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)] # compute from original frontier cell
            for (i, j) in new_frontier_cells[:]:
                if (i > (self.h-1) or i < 0 or j > (self.w-1) or j < 0) or grid[i][j] == 1:
                    new_frontier_cells.remove((i, j))
            frontier_cells = frontier_cells + new_frontier_cells
            frontier_cells.remove((x, y))

        return grid
#

    def create_maze(self):
        grid = self.generate_random_grids()
        # start and exits are in the middle range of the side and must be accessible
        # start is in the lower side
        exit0 = random.randrange(3, self.w-2)
        if grid[self.h - 1][exit0] == 0:
            for i in range(5):
                #set 5 of the cells as carpet, start is in the middle
                grid[self.h - 1][exit0-2+i] = 1
        exit0 = (self.h - 1, exit0)

        #exit1
        exit1 = random.randrange(3, self.w-2)
        if grid[0][exit1] == 0:
            for i in range(5):
                #set 5 of the cells as 1
                grid[0][exit1-2+i] = 1
        exit1 = (0, exit1)

        #exit2 is on the left side
        exit2 = random.randrange(3, self.h-2)
        if grid[exit2][0] == 0:
            for i in range(5):
                #set 5 of the cells as 1
                grid[exit2-2+i][0] = 1

        exit2 = (exit2, 0)

        #exit3 is on the right side
        exit3 = random.randrange(3, self.h-2)
        if grid[exit3][self.w-1] == 0:
            for i in range(5):
                #set 5 of the cells as 1
                grid[exit3-2+i][self.w-1] = 1
        exit3 = (exit3, self.w-1)

        self.key_grid = [exit0, exit1, exit2, exit3]
        self.grid = grid;


    def generate_xml(self):
        str_xml = self.init_str()
        grid = self.get_grid()
        key_cells = self.get_key_cells()

        for i in range(self.h):
            for j in range(self.w):
                if grid[i][j] == 1:
                    carpet_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="carpet"/>\n'.format(j+1, j+1, i+1, i+1)
                    #there are 50% chance that blocks under carpet become netherrack
#                    temp = random.randint(0, 1)
#                    if temp == 0:
                    nether_str = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="226" z2="{3}" type="netherrack"/>\n'.format(
                            j + 1, j + 1, i + 1, i + 1)

                    str_xml = str_xml + carpet_str + nether_str

        for k in range(0, len(key_cells)-1):
            i = key_cells[k][0]
            j = key_cells[k][1]
            key_grid_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="229" z2="{3}" type="emerald_block"/>\n'.format(j + 1, j + 1, i + 1, i + 1)
            str_xml = str_xml + key_grid_str
        # add fire block
        key_grid_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="fire"/>\n'.format(1,1,1,1)
        str_xml = str_xml + key_grid_str



        return str_xml

    ## only one exit at this moment for 5*5 and I'll randomly select one exit
    def generate_xml_grid(self, grid):
        
        str_xml = self.init_str()
        key_cells = self.get_key_cells()
        
        for i in range(self.h):
            for j in range(self.w):
                if grid[i][j] == 1:
                    carpet_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="carpet"/>\n'.format(j+1, j+1, i+1, i+1)
                    #there are 50% chance that blocks under carpet become netherrack
                    #                    temp = random.randint(0, 1)
                    #                    if temp == 0:
                    nether_str = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="226" z2="{3}" type="netherrack"/>\n'.format(
                                                                                                                                  j + 1, j + 1, i + 1, i + 1)
                                                                                                                                  
                    str_xml = str_xml + carpet_str + nether_str
    
        key_grid_str = ""
        while (len(key_cells) != 0):
            exit_cell = random.choice(key_cells)
            key_cells.remove(exit_cell)
            i = exit_cell[0]
            j = exit_cell[1]
            if i != 0: #can not be the same position as the start
                key_grid_str += '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="229" z2="{3}" type="emerald_block"/>\n'.format(j + 1, j + 1, i + 1, i + 1)
            else:
                continue
        key_grid_str += '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="fire"/>\n'.format(1,1,1,1)
        str_xml = str_xml + key_grid_str

        beacon_str_1 = '<DrawCuboid x1="0" y1="227" z1="{1}" x2="{0}" y2="246" z2="{1}" type="beacon"/>\n'.format(self.w+1, self.h+1)
        beacon_str_2 = '<DrawCuboid x1="0" y1="227" z1="{1}" x2="{0}" y2="246" z2="{1}" type="beacon"/>\n'.format(self.w+1, self.h+1)

        beacon_str_3 = '<DrawCuboid x1="0" y1="227" z1="{1}" x2="{0}" y2="246" z2="{1}" type="beacon"/>\n'.format(self.w+1, self.h+1)

        beacon_str_4 = '<DrawCuboid x1="0" y1="227" z1="{1}" x2="{0}" y2="246" z2="{1}" type="beacon"/>\n'.format(self.w+1, self.h+1)

        beacon_str = beacon_str_1 + beacon_str_2 + beacon_str_3 + beacon_str_4

        str_xml = str_xml + beacon_str

        return str_xml

    def get_start_xml(self):
        key_cells = self.get_key_cells()
        start_grid = key_cells[3]
        start_xml = '<Placement x="{0}" y="227" z="{1}" yaw="0"/>'.format(start_grid[1]+1, start_grid[0]+1)
        return start_xml

############get all rotated maps part#####################


    def generate_all_xml(self):
        grid = self.get_grid()
        self.grids.append(grid)  # first one.
        other_grids = self.rotate_grid(grid)
        self.grids.extend(other_grids)

        xml = []
        for i in range(len(self.grids)):
            xml.append(self.generate_xml_grid(self.grids[i]))
        return xml

    def generate_all_with_grid(self, grid):
        self.grids = []
        self.grids.append(grid)  # first one.
        other_grids = self.rotate_grid(grid)
        self.grids.extend(other_grids)

        xml = []
        for i in range(len(self.grids)):
            xml.append(self.generate_xml_grid(self.grids[i]))
        return xml

    def rotate_grid(self, grid):
        ###upside down. start == exit0
        grid1 = []
        for i in reversed(range(len(grid))):
            grid1.append(grid[i])

        ###clockwise
        grid2 = []
        for i in range(len(grid[0])):
            row = []
            for j in reversed(range(len(grid))):
                row.append(grid[j][i])
            grid2.append(row)

        ###counter clockwise
        grid3 = []
        for i in reversed(range(len(grid[0]))):
            row = []
            for j in range(len(grid)):
                row.append(grid[j][i])
            grid3.append(row)

        return [grid1, grid2, grid3]

