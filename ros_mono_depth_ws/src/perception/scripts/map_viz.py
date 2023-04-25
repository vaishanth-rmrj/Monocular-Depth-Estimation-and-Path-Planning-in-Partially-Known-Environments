#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion


class UserInterface:
    def __init__(self):
        self.map_size = (0, 0)
        map_server_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_server_callback)  

        self.map = np.array([])

    
    def map_server_callback(self, map_msg):
        print("fetching map data")
        map_width, map_height = map_msg.info.width, map_msg.info.height
        
        map = np.array(map_msg.data).reshape((map_height, map_width))
        map = np.where(map == -1, 255, map)
        map = np.where(map == 0, 150, map)
        map = np.where(map == 100, 0, map)

        self.map = map

        

    def get_map(self):
        return self.map
        


if __name__ == "__main__":

    try:
        rospy.init_node('map_viz_node')
        interface = UserInterface()

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rospy.loginfo("map viz info")
            map_image = interface.get_map()
            # print(map_image)

            if np.any(map_image):
                print(map_image)
                # cv2.imwrite("map_grey_2.png", map_image)
                cv2.imshow("Map", np.array(map_image, dtype = np.uint8 ))
                cv2.waitKey(10)
                # if cv2.waitKey(10) == ord('q'):
                #     break
            rate.sleep()
        cv2.destroyAllWindows()

    except rospy.ROSInterruptException:
        rospy.loginfo("Map viz stopped")