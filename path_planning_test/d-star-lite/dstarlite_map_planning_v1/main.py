import cv2
import numpy as np

from dstarlite import DStarLite
from grid import MapGrid

# Mouse callback function
def get_click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Click point: ({}, {})'.format(x, y))

if __name__ == "__main__":

    # map details
    # dimension: 41 x 35 m / 820x700px 
    # resolution: 0.05m/px
    map_img = cv2.imread("assets/Map_2.jpg", 0) 
    map_img = np.where(map_img > 150, 1, 0)
    map_img = np.array(map_img, dtype=np.uint8)
    # map_img *= 255
    # cv2.imshow("map", map_img.astype(np.uint8))
    # # Set the mouse callback function to select goal
    # cv2.setMouseCallback('map', get_click_point)
    # cv2.waitKey(0)

    
    map_obj = MapGrid(map_img, inflation_radius=15)
    robot_curr_pose = (124, 247)
    goal_pos = (217, 57)

    print("Start point: {} and End point: {} ".format(robot_curr_pose, goal_pos))

    # planning path based on map, start, goal
    path_planner = DStarLite(map_obj, robot_curr_pose, goal_pos)
    planned_path = path_planner.move_and_replan()