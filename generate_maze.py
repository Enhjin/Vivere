import random
import argparse
random.seed()

def make_init_maze(w, h):
    init_xml_1 = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="246" z2="{3}" type="netherrack"/>\n'.format(0, w+1, 0, h+1)
    init_xml_2 = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="246" z2="{3}" type="air"/>\n'.format(1, w, 1, h)
    init_xml_3 = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="226" z2="{3}" type="emerald_block"/>\n'.format(1, w, 1, h)
    return init_xml_1 + init_xml_2 + init_xml_3


def make_maze(w, h): #randomized_prim algorithms from wikipedia
    #1. start with a grid full of walls
    grids = [[0] * w for _ in range(h)]  # generate records, 0 means a block, 1 means a carpet
    # grids = [[0] * (w + 2)] + grids + [[0] * (w + 2)]  # surrounding grids with 0's

    assert h >= 5
    assert w >= 5
    #2. pick a cell,mark it as part of the passage.
    start_x = random.randrange(2, h-2)
    start_y = random.randrange(2, w-2)

    grids[start_x][start_y] = 1 # 1 means carpet, a part of the maze
    frontier_grids = [(start_x - 2, start_y), (start_x, start_y + 2), (start_x + 2, start_y), (start_x, start_y - 2)]
    for (i, j) in frontier_grids[:]:
        if (i > (h - 1) or i < 0 or j > (w - 1) or j < 0) or grids[i][j] == 1:  #within the grid, cannot be carpet
            frontier_grids.remove((i, j))

    while(len(frontier_grids) != 0):#While there are walls in the list:

        (x, y) = random.choice(frontier_grids) # Pick a random wall from the list.
        #get neighbor with a distance of two of the frontier cell
        neighbors_grids = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)] # in the beginning, there must have one being 1

        for (i, j) in neighbors_grids[:]:
            if (i > (h - 1) or i < 0 or j > (w - 1) or j < 0) or grids[i][j] == 0:
                neighbors_grids.remove((i, j))

        (a, b) = random.choice(neighbors_grids) #choose one neighbor
        (c, d) = ((x+a)//2, (y+b)//2)

        grids[x][y] = 1 #frontier cell becomes a part of path
        grids[c][d] = 1 #cell in the middle is a part of path
        new_frontier_grids = [(x - 2, y), (x, y + 2), (x + 2, y), (x, y - 2)] # compute from original frontier cell
        for (i, j) in new_frontier_grids[:]:
            if (i > (h-1) or i < 0 or j > (w-1) or j < 0) or grids[i][j] == 1:
                new_frontier_grids.remove((i, j))
        frontier_grids = frontier_grids + new_frontier_grids
        frontier_grids.remove((x, y))

    return grids


def adjusted_maze(w, h):
    grids = make_maze(w, h)
    # start and exits are in the middle range of the side and must be accessible
    # start is in the lower side
    start = random.randrange(2, w-2)
    if grids[h - 1][start] == 0:
        for i in range(5):
            #set 5 of the cells as carpet, start is in the middle
            grids[h - 1][start-2+i] = 1
    start = (h - 1, start)

    #exit1
    exit1 = random.randrange(2, w-2)
    if grids[0][exit1] == 0:
        for i in range(5):
            #set 5 of the cells as 1
            grids[0][exit1-2+i] = 1
    exit1 = (0, exit1)

    #exit2 is on the left side
    exit2 = random.randrange(2, h-2)
    if grids[exit2][0] == 0:
        for i in range(5):
            #set 5 of the cells as 1
            grids[exit2-2+i][0] = 1

    exit2 = (exit2, 0)

    #exit3 is on the right side
    exit3 = random.randrange(2, h-2)
    if grids[exit3][w-1] == 0:
        for i in range(5):
            #set 5 of the cells as 1
            grids[exit3-2+i][w-1] = 1
    exit3 = (exit3, w-1)

    return grids, [start, exit1, exit2, exit3]

# print('\n'.join('\t'.join('{:3}'.format(item) for item in row) for row in grids))


def grids_to_str(grids):
    str_xml = ""
    h = len(grids)
    w = len(grids[0])
    for i in range(h):
        for j in range(w):
            if grids[i][j] == 1:
                carpet_str = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="226" z2="{3}" type="carpet"/>\n'.format(j+1, j+1, i+1, i+1)
                str_xml = str_xml + carpet_str

    return str_xml

def add_key_grid(key_grids):
    str_xml = ""
    for (i, j) in key_grids:
        key_grid_str = '<DrawCuboid x1="{0}" y1="226" z1="{2}" x2="{1}" y2="226" z2="{3}" type="sea_lantern"/>\n'.format(j + 1,
                                                                                                                  j + 1,
                                                                                                                  i + 1,
                                                                                                                 i + 1)
        str_xml = str_xml + key_grid_str
    #add fire block
    # key_grid_str = '<DrawCuboid x1="{0}" y1="227" z1="{2}" x2="{1}" y2="227" z2="{3}" type="fire"/>\n'.format(1,1,1,1)

    str_xml = str_xml + key_grid_str
    return str_xml

def maze_to_str(w, h):
    w = int(w)
    h = int(h)
    grids, key_grids = adjusted_maze(w, h)
    str_init = make_init_maze(w, h)
    str_grids = grids_to_str(grids)
    str_key_grids = add_key_grid(key_grids)
    str_all = str_init + str_grids + str_key_grids
    print(str_all)
    return str_all

if __name__ == '__main__':
    desc = "w >= 5, h >= 5"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "-w", "--width", required=True,
        help="the width of the map"
    )

    parser.add_argument(
        "-d", "--height", required=True,
        help="the height of the map"
    )

    args = parser.parse_args()

    maze_to_str(args.width, args.height)
