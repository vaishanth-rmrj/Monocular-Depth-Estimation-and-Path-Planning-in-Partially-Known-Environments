#!/usr/bin/env python3

import rospy
from geometry_msgs.msg  import Twist
from sensor_msgs.msg import Joy

class JoyTeleop:
    def __init__(self):
        self.max_speed = 0.8
        self.curr_speed = 0.1
        # trigger btn index
        # square btn = 3
        # cross btn = 0
        # circle btn = 1
        # triange btn = 2
        self.trigger_btn = 1
        # movement btn id
        self.forw_and_back_btn = 7
        self.left_and_right_btn = 6

        self.twist = Twist()

        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    
    def joy_callback(self, joy_msg):

        self.btns = joy_msg.buttons
        self.axes= joy_msg.axes
        # print(self.axes)

        # velocity control
        if self.btns[2] and self.curr_speed < self.max_speed:
            self.curr_speed *= 1.3
        
        if self.btns[0] and self.curr_speed > 0.1:
            self.curr_speed *= 0.7
        
        if self.btns[self.trigger_btn]:
            # forward and backbard motion control
            if self.axes[self.forw_and_back_btn] == 1:
                self.twist.linear.x = self.curr_speed
                self.twist.angular.z = 0.0
            elif self.axes[self.forw_and_back_btn] == -1:
                self.twist.linear.x = -self.curr_speed
                self.twist.angular.z = 0.0
            else:
                pass

            # left and right motion control
            if self.axes[self.left_and_right_btn] == 1.0:
                self.twist.linear.x = 0.0
                self.twist.angular.z = self.curr_speed
            elif self.axes[self.left_and_right_btn] == -1.0:
                self.twist.linear.x = 0.0
                self.twist.angular.z = -self.curr_speed
            else:
                pass
            
            # hault motion when no btn pressed
            if self.axes[self.forw_and_back_btn] == 0 and self.axes[self.left_and_right_btn] == 0:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
        
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0


        # cmd publish
        self.cmd_vel_pub.publish(self.twist)


        


if __name__ == "__main__":
    rospy.init_node('joy_teleop', anonymous=True)
    joy_teleop = JoyTeleop()
    rospy.spin()
