#!/usr/bin/env python3

import time
import kivy
import rospy
import threading

# kivy imports
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.utils import get_color_from_hex
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button

from kivy.clock import Clock
from kivy.properties import StringProperty, ObjectProperty, NumericProperty

from functools import partial

# ros imports
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

# colors used
#  dark indigo - rgba(63, 81, 181, 1)            

       
    
class ControlScreen(Screen):   

    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self, 'text')
        if self._keyboard.widget:
            # If it exists, this widget is a VKeyboard object which you can use
            # to change the keyboard layout.
            pass       
        self._keyboard.bind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)

        # ros params
        self.twist_msg = Twist()
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1, latch=True)
        self.max_vel = 0.8
        self.vel_scale = 0.1

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        self._keyboard = None
    
    def unpress(self, *args):
        self.ids.manual_up_btn.md_bg_color = "green"
        self.ids.manual_up_btn.pos_hint = {'center_y':.75}

        self.ids.manual_left_btn.md_bg_color = "green"
        self.ids.manual_left_btn.pos_hint = {'center_x':.25}

        self.ids.manual_right_btn.md_bg_color = "green"
        self.ids.manual_right_btn.pos_hint = {'center_x':.75}

        self.ids.manual_down_btn.md_bg_color = "green"
        self.ids.manual_down_btn.pos_hint = {'center_y':.25}

        self.stop_motion()

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':
            print("Pressed W")
            self.move_fwd()
            self.ids.manual_up_btn.md_bg_color = "blue"
            self.ids.manual_up_btn.pos_hint = {'center_y':.76}

        elif keycode[1] == 's':
            print("Pressed s")
            self.move_bwd()
            self.ids.manual_down_btn.md_bg_color = "blue"
            self.ids.manual_down_btn.pos_hint = {'center_y':.24}

        elif keycode[1] == 'a':
            print("Pressed a")
            self.move_left()
            self.ids.manual_left_btn.md_bg_color = "blue"
            self.ids.manual_left_btn.pos_hint = {'center_x':.24}

        elif keycode[1] == 'd':
            print("Pressed d")
            self.move_right()
            self.ids.manual_right_btn.md_bg_color = "blue"
            self.ids.manual_right_btn.pos_hint = {'center_x':.76}

        return True
    
    def _on_keyboard_up(self, keyboard, keycode):
        print("Key is up ")
        self.unpress()
        return True
    
    def control_btn_pressed(self, btn_type):
        if btn_type == 'w':
            print("Pressed W")
            self.move_fwd()
            self.ids.manual_up_btn.md_bg_color = "blue"
            self.ids.manual_up_btn.pos_hint = {'center_y':.76}

        elif btn_type == 's':
            print("Pressed s")
            self.move_bwd()
            self.ids.manual_down_btn.md_bg_color = "blue"
            self.ids.manual_down_btn.pos_hint = {'center_y':.24}

        elif btn_type == 'a':
            print("Pressed a")
            self.move_left()
            self.ids.manual_left_btn.md_bg_color = "blue"
            self.ids.manual_left_btn.pos_hint = {'center_x':.24}

        elif btn_type == 'd':
            print("Pressed d")
            self.move_right()
            self.ids.manual_right_btn.md_bg_color = "blue"
            self.ids.manual_right_btn.pos_hint = {'center_x':.76}
    
    def control_btn_up(self):
        print("Key is up")
        self.unpress()
    
    def update_velocity_scale(self, scale_dir):
        if scale_dir == 1:
            if self.vel_scale < 1:
                self.vel_scale += 0.1
            else:
                print("Max velocity limit reached !")
        else:
            if self.vel_scale > 0.1:
                self.vel_scale -= 0.1
            else:
                print("Min velocity limit reached !")
        self.vel_scale = round(self.vel_scale, 2)
        self.ids.velocity_scale_label.text = str(int(self.vel_scale * 100)) + "%"

    # ros interface funcs
    def move_fwd(self):
        self.twist_msg.linear.x = round(self.max_vel * self.vel_scale, 2)
        self.twist_msg.angular.z = 0
        self.pub_vel_cmd()

    def move_bwd(self):
        self.twist_msg.linear.x = -round(self.max_vel * self.vel_scale, 2)
        self.twist_msg.angular.z = 0
        self.pub_vel_cmd()

    def move_right(self):
        self.twist_msg.linear.x = 0
        self.twist_msg.angular.z = -round(self.max_vel * self.vel_scale, 2)
        self.pub_vel_cmd()

    def move_left(self):
        self.twist_msg.linear.x = 0
        self.twist_msg.angular.z = round(self.max_vel * self.vel_scale, 2)
        self.pub_vel_cmd()
    
    def stop_motion(self):
        self.twist_msg.linear.x = 0
        self.twist_msg.angular.z = 0
        self.pub_vel_cmd()

    def pub_vel_cmd(self):
        print("Publishing command vel")
        self.cmd_pub.publish(self.twist_msg)

# utilities


class MyButton(Button):
    #add these three properties in the class
    icon = ObjectProperty(None)
    icon_size = (0,0)
    icon_padding = NumericProperty(0)


class RobotInterfaceApp(App):

    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
        # self.theme_cls.theme_style = "Light"
        self.kv = Builder.load_file("assets/interface.kv")
         

    def build(self):    
        self.title = "Interface"        
        sm = ScreenManager()
        sm.add_widget(ControlScreen(name='control_screen'))
        return sm

if __name__ == "__main__":
    rospy.init_node("bot_interface", anonymous=True)
    client_thread = threading.Thread(target=RobotInterfaceApp().run(),daemon=True)
    client_thread.start()
    