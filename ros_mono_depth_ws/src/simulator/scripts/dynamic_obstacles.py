#!/usr/bin/env python
import rospy
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates

class DynamicObstacles:
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self):
        while not rospy.is_shutdown():
            obstacle = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            print(model)
            print("moving obstacles")
            for i in range(len(model.name)):
                if model.name[i] == 'dyn_obst_1':
                    obstacle.model_name = 'dyn_obst_1'
                    obstacle.pose = model.pose[i]
                    obstacle.twist = Twist()
                    obstacle.twist.angular.z = 0.5
                    self.pub_model.publish(obstacle)
                    print("moving dynamic obstacle 1")
                    time.sleep(0.1)


                if model.name[i] == 'dyn_obst_2':
                    obstacle.model_name = 'dyn_obst_2'
                    obstacle.pose = model.pose[i]
                    obstacle.twist = Twist()
                    obstacle.twist.angular.z = 0.5
                    self.pub_model.publish(obstacle)
                    print("moving dynamic obstacle 2")
                    time.sleep(0.1)

                if model.name[i] == 'dyn_obst_3':
                    obstacle.model_name = 'dyn_obst_3'
                    obstacle.pose = model.pose[i]
                    obstacle.twist = Twist()
                    obstacle.twist.angular.z = 0.5
                    print(obstacle)
                    self.pub_model.publish(obstacle)
                    print("moving dynamic obstacle 3")
                    time.sleep(0.1)

def main():
    rospy.loginfo("Dynamic obstacle node started :)")
    rospy.init_node('moving_obstacle')
    moving = DynamicObstacles()

if __name__ == '__main__':
    print("started")
    main()