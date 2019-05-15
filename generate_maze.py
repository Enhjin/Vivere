import random
import argparse
random.seed()

class Maze:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.key_grid = []
        self.grids = []

    def init_str(self):
        init_xml_1 = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="246" z2="{3}" type="netherrack"/>\n'.format(0, self.w+1, 0, self.h+1)
        init_xml_2 = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="246" z2="{3}" type="air"/>\n'.format(1, self.w, 1, self.h)
        init_xml_3 = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="sea_lantern"/>\n'.format(1, self.w, 1, self.h)
        return init_xml_1 + init_xml_2 + init_xml_3

    def get_grids(self):
        if len(self.grids) == 0:
            self.create_maze()
        return self.grids

    def get_key_grids(self):
        if len(self.key_grid) == 0:
            self.create_maze()
        return self.key_grid


    def generate_random_grids(self): #randomized_prim algorithms from wikipedia
        #1. start with a grid full of walls
        grids = [[0] * self.w for _ in range(self.h)]  # generate records, 0 means a block, 1 means a carpet
        # grids = [[0] * (w + 2)] + grids + [[0] * (w + 2)]  # surrounding grids with 0's

        assert self.h >= 5
        assert self.w >= 5
        #2. pick a cell,mark it as part of the passage.
        start_x = random.randrange(2, self.h-2)
        start_y = random.randrange(2, self.w-2)

        grids[start_x][start_y] = 1 # 1 means carpet, a part of the maze
        frontier_grids = [(start_x - 2, start_y), (start_x, start_y + 2), (start_x + 2, start_y), (start_x, start_y - 2)]
        for (i, j) in frontier_grids[:]:
            if (i > (self.h - 1) or i < 0 or j > (self.w - 1) or j < 0) or grids[i][j] == 1:  #within the grid, cannot be carpet
                frontier_grids.remove((i, j))

        while(len(frontier_grids) != 0):#While there are walls in the list:

            (x, y) = random.choice(frontier_grids) # Pick a random wall from the list.
            #get neighbor with a distance of two of the frontier cell
            neighbors_grids = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)] # in the beginning, there must have one being 1

            for (i, j) in neighbors_grids[:]:
                if (i > (self.h - 1) or i < 0 or j > (self.w - 1) or j < 0) or grids[i][j] == 0:
                    neighbors_grids.remove((i, j))

            (a, b) = random.choice(neighbors_grids) #choose one neighbor
            (c, d) = ((x+a)//2, (y+b)//2)

            grids[x][y] = 1 #frontier cell becomes a part of path
            grids[c][d] = 1 #cell in the middle is a part of path
            new_frontier_grids = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)] # compute from original frontier cell
            for (i, j) in new_frontier_grids[:]:
                if (i > (self.h-1) or i < 0 or j > (self.w-1) or j < 0) or grids[i][j] == 1:
                    new_frontier_grids.remove((i, j))
            frontier_grids = frontier_grids + new_frontier_grids
            frontier_grids.remove((x, y))

        #finished generating grids
        # self.grids = grids

        return grids


    def create_maze(self):
        grids = self.generate_random_grids()
        # start and exits are in the middle range of the side and must be accessible
        # start is in the lower side
        start = random.randrange(2, self.w-2)
        if grids[self.h - 1][start] == 0:
            for i in range(5):
                #set 5 of the cells as carpet, start is in the middle
                grids[self.h - 1][start-2+i] = 1
        start = (self.h - 1, start)

        #exit1
        exit1 = random.randrange(2, self.w-2)
        if grids[0][exit1] == 0:
            for i in range(5):
                #set 5 of the cells as 1
                grids[0][exit1-2+i] = 1
        exit1 = (0, exit1)

        #exit2 is on the left side
        exit2 = random.randrange(2, self.h-2)
        if grids[exit2][0] == 0:
            for i in range(5):
                #set 5 of the cells as 1
                grids[exit2-2+i][0] = 1

        exit2 = (exit2, 0)

        #exit3 is on the right side
        exit3 = random.randrange(2, self.h-2)
        if grids[exit3][self.w-1] == 0:
            for i in range(5):
                #set 5 of the cells as 1
                grids[exit3-2+i][self.w-1] = 1
        exit3 = (exit3, self.w-1)

        self.key_grid = [start, exit1, exit2, exit3]
        self.grids = grids;


    def generate_xml(self):
        str_xml = self.init_str()
        grids = self.get_grids()
        key_grids = self.get_key_grids()

        for i in range(self.h):
            for j in range(self.w):
                if grids[i][j] == 1:
                    carpet_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="carpet"/>\n'.format(j+1, j+1, i+1, i+1)
                    str_xml = str_xml + carpet_str

        for k in range(0, len(key_grids)-1):
            i = key_grids[k][0]
            j = key_grids[k][1]
            key_grid_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="226" z2="{3}" type="emerald_block"/>\n'.format(j + 1, j + 1, i + 1, i + 1)
            str_xml = str_xml + key_grid_str
        # add fire block
        key_grid_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="fire"/>\n'.format(1,1,1,1)
        str_xml = str_xml + key_grid_str

        return str_xml

    def get_start_xml(self):
        key_grids = self.get_key_grids();
        start_grid = key_grids[3]
        start_xml = '<Placement x="{0}" y="227" z="{1}" yaw="0"/>'.format(start_grid[1]+1, start_grid[0]+1)
        return start_xml
