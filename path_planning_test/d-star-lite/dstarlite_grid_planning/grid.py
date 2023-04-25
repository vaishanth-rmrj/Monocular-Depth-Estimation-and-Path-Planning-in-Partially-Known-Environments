import numpy as np

WALL = '#'
PASSABLE = '.'

class MapGrid:
    def __init__(self, grid_string="", width=0, height=0):       
        
        self.walls = set()
        self.width = width
        self.height = height
        self.map_img = np.ones((height, width))

        # either set map shape from string or values
        if len(grid_string) > 0:
            self.width, self.height, self.start, self.goal, self.map_img = self.grid_from_string(grid_string)
        
    
    def grid_from_string(self, grid_string):
        """
        Construct a SquareGrid from a string representation
        Representation:
        . - a passable square
        A - the start position
        Z - the goal position
        # - an unpassable square (a wall)
        Args:
            :type string: str
        Returns a 3-tuple: (g: SquareGrid, start: Tuple, end: Tuple)
        """
        assert grid_string.count('A') == 1, "Cant have more than 1 start position!"
        assert grid_string.count('Z') == 1, "Cant have more than 1 end position!"
        
        lines = [l.strip() for l in grid_string.split('\n') if l.strip()]

        width, height = len(lines[0]), len(lines)
        start, goal = None, None
        map_img = np.ones((height, width))
        # loop thru the stings to fetch walls, start and goal pt
        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                if char == WALL:
                    map_img[row, col] = 0
                    self.walls.add((col, row))
                if char == 'A':
                    start = (col, row)
                if char == 'Z':
                    goal = (col, row)
        assert start is not None
        assert goal is not None

        return width, height, start, goal, map_img

    

    def cost(self, from_node, to_node):
        if from_node in self.walls or to_node in self.walls:
            return float('inf')
        else:
            return 1

    def neighbors(self, vertex, no_move_direction=4):
        (x, y) = vertex
        if no_move_direction == 4:
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if no_move_direction == 8:
            results = [(x + 1, y), (x + 1, y - 1), (x, y - 1), (x -1, y - 1), 
                       (x - 1, y), (x - 1 , y + 1), (x, y+1), (x + 1, y + 1)]

        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        return list(results)
    
    def in_bounds(self, vertex):
        '''
        to check if vertex is in map bounds
        '''
        (x, y) = vertex
        return 0 <= x < self.width and 0 <= y < self.height

    def get_local_region(self, pos, view_dist=2):
        '''
        to fetch the local reg surrounding the curr pos

        return: node: 0 or 1
        0 - wall
        1 - freespace
        '''
        (px, py) = pos
        local_nodes = [(x, y) for x in range(px - view_dist, px + view_dist + 1)
                        for y in range(py - view_dist, py + view_dist + 1)
                        if self.in_bounds((x, y))]
        
        return {node: 0 if node in self.walls else 1 for node in local_nodes}


class ExploredGrid(MapGrid):

    def get_unexplored_walls(self, local_reg):
        '''
        return: walls not explored already
        '''
        new_detected_walls = {node for node, node_type in local_reg.items() if node_type == 0}        
        return new_detected_walls - self.walls

    def update_walls(self, new_walls):

        # updating new walls to explored map
        for wall in new_walls:
            wall_x, wall_y = wall
            self.map_img[(wall_y, wall_x)] = 0

        self.walls.update(new_walls)