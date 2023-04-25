import cv2
import numpy as np

from utility import convert_grid_to_img
from dstarlite import DStarLite
from grid import MapGrid

def load_map_from_file(grid_path):
    grid_in_string = ""
    with open(grid_path, 'r') as f:
        for line in f.readlines():
            grid_in_string += line
    return grid_in_string

if __name__ == "__main__":

    grid_in_string = load_map_from_file('custom_map.txt')

    # converting sting to grid map object
    map_grid = MapGrid(grid_in_string)
    start_pt = map_grid.start
    goal_pt = map_grid.goal    

    print("Start point: {} and End point: {} ".format(start_pt, goal_pt))

    # planning path based on map, start, goal
    path_planner = DStarLite(map_grid, start_pt, goal_pt)
    planned_path = path_planner.move_and_replan()



