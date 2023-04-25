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
from kivy.graphics.texture import Texture
from kivy_garden.graph import Graph, MeshLinePlot

from functools import partial

class MapWidget(Widget):

    def on_map_touch(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        print(*touch.pos)
    

class MapScreen(Screen):
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)    

        self.name = "map_screen"   

    def touch_event(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        scatter_ = self.ids.scatter
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if scatter_.scale < 20:
                    scatter_.scale = scatter_.scale * 1.1
            elif touch.button == 'scrollup':
                if scatter_.scale > 1:
                    scatter_.scale = scatter_.scale * 0.8

        elif self.ids.my_image.collide_point(*touch.pos):
            print(self.ids.my_image)
            touch_pos = self.ids.my_image.to_local(*touch.pos)
            print(touch_pos)
        
        # elif self.collide_point(*touch.pos):
        #     print(*touch.pos)
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            print("nothing happen")
    



class MapApp(App):
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
        # self.theme_cls.theme_style = "Light"
        self.kv = Builder.load_file("map_gui_style.kv") 

    def build(self):    
        self.title = "Map GUI"        
        sm = ScreenManager()
        sm.add_widget(MapScreen())
        return sm

if __name__ == "__main__":
    MapApp().run()