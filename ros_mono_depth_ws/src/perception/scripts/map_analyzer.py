#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf import TransformListener, ExtrapolationException, LookupException
from tf2_msgs.msg import TFMessage
import tf2_ros
from tf.transformations import euler_from_quaternion


class Map:
    def __init__(self):
        self.frame_width = 900
        self.frame_height = 600
        self.map_frame = np.zeros((self.frame_height, self.frame_width))
        self.transform_listener = TransformListener()
        self.robot_pose = []

        rospy.Subscriber("/map", OccupancyGrid, self.map_server_callback)  
        # rospy.Subscriber("/tf", TFMessage, self.tf_callback)

        self.width_scaling, self.height_scaling = 0, 0
        
    # def tf_callback(self, tf_data):
    #     try:
    #         pos, quaternion = self.transform_listener.lookupTransform("/base_footprint", "/map", rospy.Time())
    #         # print(self.robot_pose)
    #     except (ExtrapolationException, LookupException):
    #         return
        
    #     print(pos.transform.translation.x)

    def tf_callback(self):
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        try:
            trans = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(), rospy.Duration(2))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            rot_q = trans.transform.rotation
            self.robot_pose = [x, y]
            (roll, pitch, yaw) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("error")
            pass

    def on_mouse_click(self, event, x, y, p1, p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Cordinated based on trasformed image:", x, y)
            # print("Cordinated based on original image:", ((int(x*self.width_scaling)), (int(y*self.height_scaling))))
            cv2.circle(self.map_frame, (x, y), 3, (0), 2)
            # cv2.circle(self.org_map_frame, ((int((self.map_frame.shape[0] - y)*self.width_scaling)), 
            #                                 (int(x*self.height_scaling))), 3, (0), 2)

            self.pt_dst_from_origin = self.transformed_map_origin_x - x, self.transformed_map_origin_y - y
            print("Point dist from origin:", self.pt_dst_from_origin)

            cv2.circle(self.org_map_frame, ((int(self.transformed_map_origin_x - x)), 
                                            (int(self.transformed_map_origin_y - y))), 3, (0), 2)
            
            self.robot_pose_map = (((self.map_frame.shape[0] - y)/self.width_scaling)-self.map_origin[0])* self.map_resolution, \
                                    ((x/self.height_scaling)-self.map_origin[1])* self.map_resolution
            # print("Actual robot pose in map: ", self.robot_pose)
            # print("Robot pose in map cordinates: ", self.robot_pose_map)
    
    def map_server_callback(self, map_msg):
        self.tf_callback()
        # print(self.robot_pose)
        
        if len(map_msg.data) and len(self.robot_pose) > 1:
            org_width, org_height = map_msg.info.width, map_msg.info.height
            self.map_origin = np.array([map_msg.info.origin.position.x, map_msg.info.origin.position.y])
            self.map_resolution = map_msg.info.resolution

            map_data = np.array(map_msg.data).reshape((org_height, org_width))
            map_data = np.where(map_data == -1, 255, map_data)
            map_data = np.where(map_data == 0, 150, map_data)
            map_data = np.where(map_data == 100, 0, map_data)
            map_data = map_data.astype(np.uint8)

            # robot pose on map img
            px = int((self.robot_pose[0] - self.map_origin[0])/self.map_resolution)
            py = int((self.robot_pose[1] - self.map_origin[1])/self.map_resolution)
            
            #  drawing robot pose on map img
            map_data = cv2.circle(map_data, (px, py), 3, (0), 5) 
            # self.org_map_frame = cv2.flip(map_data, 1)  

            print("Map image origin:", (abs(int(self.map_origin[0]/self.map_resolution)),abs(int(self.map_origin[1]/self.map_resolution))))
            map_data = cv2.circle(map_data, (abs(int(self.map_origin[0]/self.map_resolution)),abs(int(self.map_origin[1]/self.map_resolution))), 3, (0), 2)
            map_data = cv2.rotate(map_data, cv2.ROTATE_90_CLOCKWISE)      
            self.org_map_frame = cv2.flip(map_data, 1)         
            height, width = map_data.shape

            if width < self.frame_width:
                v_pad, h_pad = 0, abs(self.frame_width//2 - width//2)
                map_data = np.pad(map_data, ((v_pad, v_pad), (h_pad, h_pad)), 
                                  'constant', constant_values=(255,))
            else:
                map_data = cv2.resize(map_data, (self.frame_width, height))

            if height < self.frame_height:
                v_pad, h_pad = abs(self.frame_height//2 - height//2), 0
                map_data = np.pad(map_data, ((v_pad, v_pad), (h_pad, h_pad)), 
                                  'constant', constant_values=(255,))
            else:
                map_data = cv2.resize(map_data, (height, self.frame_height))

            self.transformed_map_origin_x = abs(int(self.map_origin[0]/self.map_resolution))+v_pad
            self.transformed_map_origin_y = abs(int(self.map_origin[1]/self.map_resolution))+h_pad
            print("Transformed Map image origin:", (self.transformed_map_origin_x, self.transformed_map_origin_y))
            
            # compunting the map scaling params
            # for debugging
            org_width, org_height = org_height, org_width

            self.width_scaling, self.height_scaling = (org_width/map_data.shape[1])*1, (org_height/map_data.shape[0])*1
            # print("scalling : ", self.width_scaling, self.height_scaling)
            # print("Org map dim", org_width, org_height)
            # print("Transformed map dim:", map_data.shape)

            # flipping the map
            self.map_frame = cv2.flip(map_data, 1)  
    
    def run(self):

        if np.any(self.map_frame):
            cv2.namedWindow("Map Display Frame")
            cv2.imshow("Map Display Frame", self.map_frame.astype(np.uint8))         
            cv2.imshow("Original Map Display Frame", self.org_map_frame.astype(np.uint8))       
            cv2.setMouseCallback("Map Display Frame", self.on_mouse_click)

            cv2.waitKey(2)    

if __name__ == "__main__":

    try:
        rospy.init_node('map_analyzer')
        map_analyzer = Map()

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            map_analyzer.run()         
            rate.sleep()
            # rospy.spin()

        cv2.destroyAllWindows()

    except rospy.ROSInterruptException:
        rospy.loginfo("Map viz stopped")
        cv2.destroyAllWindows()