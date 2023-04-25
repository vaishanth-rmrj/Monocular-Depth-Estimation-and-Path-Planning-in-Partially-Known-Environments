from grid import ExploredGrid, MapGrid
from priority_queue import PriorityQueue
from collections import deque
from functools import partial
from utility import render_to_rgb
import cv2
import numpy as np
import time

class DStarLite(object):
    def __init__(self, grid_map, start, goal, view_range=50):
        

        # Init the graphs
        # empty_map_img = np.ones(map_obj.grid_map.shape, dtype=np.float32)
        # self.explored_grid = ExploredGrid(map_img=map_obj.grid_map, convert_to_grid=False)
        # self.org_mapgrid = map_obj
        self.grid_map = grid_map
        self.view_range = view_range

        self.true_start_node = start
        self.true_curr_node = self.true_start_node
        self.true_goal_node = goal

        self.start_node = self.grid_map.get_grid_pt(start)
        self.curr_node = self.start_node
        self.goal_node = self.grid_map.get_grid_pt(goal)
        self.is_new_goal = False

        self.back_pointers = {}
        self.g_val = {}
        self.rhs_val = {}
        # set rhs val of goal to 0
        self.rhs_val[self.goal_node] = 0
        self.Km = 0
        
        self.queue = PriorityQueue()
        # inserting goal node to the queue
        self.queue.insert(vertex = self.goal_node, 
                          priority_keys = self.calc_key(self.goal_node))

        self.back_pointers[self.goal_node] = None
    
    def set_goal(self, pt):
        self.true_goal_node = pt
        self.goal_node = self.grid_map.get_grid_pt(self.true_goal_node)
        self.is_new_goal = True
        # resetting values
        self.back_pointers = {}
        self.g_val = {}
        self.rhs_val = {}
        # set rhs val of goal to 0
        self.rhs_val[self.goal_node] = 0
        self.Km = 0
        self.queue = PriorityQueue()
        # inserting goal node to the queue
        self.queue.insert(vertex = self.goal_node, 
                          priority_keys = self.calc_key(self.goal_node))

        self.back_pointers[self.goal_node] = None
        

    # Mouse callback function
    def get_click_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Click point: ({}, {})'.format(x, y))
            self.set_goal((x, y))

            # # for debugging only
            # true_map = self.org_mapgrid.true_map.copy()
            # # processing true map grid        
            # true_map = render_to_rgb(true_map)
            # # adding curr robot pose amd goal pose
            # true_robot_pose = self.org_mapgrid.get_true_pt(self.curr_node[::-1])
            # # true_goal_node = self.org_mapgrid.get_true_pt(self.goal_node[::-1])
            # cv2.circle(true_map, true_robot_pose, 2, [255, 109, 83], 5)
            # cv2.circle(true_map, (x, y), 2, [255, 109, 83], 5)
            # cv2.imshow("Map Grid 2", true_map)
            # cv2.waitKey(0)

            
    
    def calc_key(self, node_pt):
        node_g_val = self.get_gval(node_pt)
        node_rhs_val = self.get_rhsval(node_pt)

        # similar to total cost to reach final node
        key_0 = min(node_g_val, node_rhs_val) + self.calc_heuristic(node_pt, self.curr_node) + self.Km
        # similar to min cost to come to this node
        key_1 = min(node_g_val, node_rhs_val) 

        return [key_0, key_1]

    def calc_heuristic(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        # heuristic is manhatten distance
        return abs(x1-x2) + abs(y1-y2)

    def get_gval(self, node):
        return self.g_val.get(node, float('inf'))
        # return self.g_val[(node[1], node[0])]

    def get_rhsval(self, node):
        # print(self.rhs_val)
        return self.rhs_val.get(node, float('inf'))
        # return self.rhs_val[(node[1], node[0])]
    
    def calc_rhs_val(self, node):
        # print("node", node)
        lowest_cost_neighbour = self.fetch_lowest_cost_neighbour(node)
        # print("lowest cost neighbout", lowest_cost_neighbour)
        # print("Lookahead cost", self.calc_lookahead_cost(node, lowest_cost_neighbour))
        return self.calc_lookahead_cost(node, lowest_cost_neighbour), lowest_cost_neighbour

    def fetch_lowest_cost_neighbour(self, node):
        neighbours = self.grid_map.neighbors(node)
        cost_fn = partial(self.calc_lookahead_cost, node)
        lowest_cost_neighbour = min(neighbours, key=cost_fn)
        # true_lowest_cost_neighbour = self.grid_map.get_true_pt(lowest_cost_neighbour)
        # return lowest_cost_neighbour, true_lowest_cost_neighbour
        return lowest_cost_neighbour

    def calc_lookahead_cost(self, node, neighbour):
        return self.get_gval(neighbour) + self.grid_map.cost(node, neighbour)

    def update_nodes(self, nodes):
        for node in nodes:
            if node != self.goal_node:
                # setting the rhs value of the node
                # and setting the backpointer of the node
                # to the lowest cost neighbour
                # print( "Return val of calc_rhs_val",self.calc_rhs_val(node))
                self.rhs_val[node], self.back_pointers[node] = self.calc_rhs_val(node)
            
            # print("Node vertex val",node)
            # print("Node g val",self.get_gval(node))
            # print("Node rhs val",self.get_rhsval(node))
            if self.get_gval(node) != self.get_rhsval(node):
                # if the node is locally inconsistent
                # update the node keys and add to the queue
                self.queue.delete(node)
                self.queue.insert(node, self.calc_key(node))
            else:
                # if the node is locally consistent
                # del node from queue
                self.queue.delete(node)

    def compute_shortest_path(self):
        
        t_start = time.time()
        last_nodes = deque(maxlen=200)
        # loop until 2 conditions are satisfied
        # 1. current / start node is locally consistent
        # 2. there exists no node with key val less than curr node
        counter = 0        
        while self.queue.top_key() < self.calc_key(self.curr_node) or self.g_val.get(self.curr_node) != self.rhs_val.get(self.curr_node):
            
            print(counter) 
            counter += 1  
            k_old = self.queue.top_key()
            node = self.queue.pop_smallest()[1]

            # fail safe
            last_nodes.append(node)
            if len(last_nodes) == 200 and len(set(last_nodes)) < 10:
                print("Stuck in loop")
                raise Exception("Stuck in a loop")
                break

            k_new = self.calc_key(node_pt=node)
            

            if k_new > k_old:
                # nodes key value has changed because of obstacle
                self.queue.insert(node, k_new)
            elif self.get_gval(node) > self.get_rhsval(node):
                # since lookahead val of node is less than its gval
                # we can reach the node with the min val of rhs
                self.g_val[node] = self.get_rhsval(node)

                # updating neighbour node
                neighbours = self.grid_map.neighbors(node)
                self.update_nodes(neighbours)
            else:
                self.g_val[node] = float('inf')
                neighbours = self.grid_map.neighbors(node)
                self.update_nodes(neighbours+[node])
        
        elasped_time = time.time()-t_start
        if elasped_time > 1:
            print("Time taken to compute shortest path: ", (elasped_time))

    def get_path(self):
        curr_node = self.curr_node
        path = []
        while True:
            if curr_node in self.back_pointers:
                path.append(curr_node)
                curr_node = self.back_pointers[curr_node]            
            else:
                break     
        return path
    
    def render_pathplanning(self, grid_map, explored_grid_map, start_pt, curr_pt, goal_pt, path, wait_time=24):
        # print("Computed path points count: ", len(path))
        # grid_map = grid_map.copy()
        explored_grid_map = explored_grid_map.copy()

        # processing true map grid        
        # grid_map = render_to_rgb(grid_map)
        # # adding start and goal points
        # cv2.circle(grid_map, start_pt, 2, [255, 109, 83], 5)
        # cv2.circle(grid_map, goal_pt, 2, [60, 142, 56], 5)

        # processing explored map grid      
        explored_grid_map = render_to_rgb(explored_grid_map)
        # adding path points
        if len(path) > 3:
            for point in path:
                cv2.circle(explored_grid_map, point, 1, [251, 140, 0], 1)
        # adding start and goal points
        cv2.circle(explored_grid_map, curr_pt, 2, [255, 109, 83], 5)
        cv2.circle(explored_grid_map, goal_pt, 2, [60, 142, 56], 5)

        # side_by_side_img = np.concatenate((grid_map, explored_grid_map), axis=1)
        cv2.imshow("Map Grid - Explored Map Grid: (Press q for next)", explored_grid_map)
        if cv2.waitKey(wait_time) == ord('q'):
            return 0
    
    def display_map(self, show_path=False):
        true_map = self.grid_map.true_map.copy()
        # processing true map grid        
        true_map = render_to_rgb(true_map)
        # adding curr robot pose amd goal pose
        cv2.circle(true_map, self.true_curr_node, 2, [255, 109, 83], 5)
        if self.true_goal_node: cv2.circle(true_map, self.true_goal_node, 2, [255, 109, 83], 5)

        # rendering original path
        if show_path:
            original_path=[self.grid_map.get_true_pt(path_pt) for path_pt in self.get_path()]
            
            for org_path_pt in original_path:
                cv2.circle(true_map, org_path_pt, 1, [251, 140, 0], 1)

        cv2.imshow("Map Grid", true_map)
        # Set the mouse callback function
        cv2.setMouseCallback("Map Grid", self.get_click_point)

        if cv2.waitKey(24) == ord('q'):
            return 0
            
    def move_and_replan(self):

        # get local region for the curr node
        local_region_pixels, grid_pixels, true_map_pixel_index, grid_arr_index, self.curr_node = self.grid_map.get_local_region(self.true_curr_node, self.view_range)
        obs_pts = self.grid_map.update_explored_grid(grid_pixels, true_map_pixel_index, grid_arr_index,  self.curr_node)
        unexplored_obs = self.grid_map.get_unexplored_obstacles(obs_pts)
        self.grid_map.update_obstacles(unexplored_obs)

        # compute shortest path to goal traverses from goal node to start node
        # this path is with the explored obstacles
        self.compute_shortest_path()
        
        # for debugging
        # self.render_pathplanning(
        #     grid_map=self.grid_map.true_map,
        #     explored_grid_map=self.grid_map.explored_grid_map,
        #     start_pt=self.start_node,
        #     curr_pt=self.curr_node,
        #     goal_pt=self.goal_node,
        #     path=self.get_path()
        # )
        
        last_node = self.curr_node
        iter = 0
        while self.curr_node != self.goal_node:  
            iter += 1          

            # if self.get_gval(self.curr_node) == float('inf'):
            #     raise Exception("no path found")

            self.curr_node = self.fetch_lowest_cost_neighbour(self.curr_node)            
            self.true_curr_node = self.grid_map.get_true_pt(self.curr_node)

            # get local region for the curr node
            local_region_pixels, grid_pixels, true_map_pixel_index, grid_arr_index, self.curr_node = self.grid_map.get_local_region(self.true_curr_node, self.view_range)
            obs_pts = self.grid_map.update_explored_grid(grid_pixels, true_map_pixel_index, grid_arr_index,  self.curr_node)
            unexplored_obs = self.grid_map.get_unexplored_obstacles(obs_pts)
            
            if unexplored_obs:
                self.grid_map.update_obstacles(unexplored_obs)
                if iter%10 == 0: self.grid_map.update_grid_inflation_and_costmap()
                # cv2.imshow("self.explored_grid_costmap", (self.grid_map.explored_grid_costmap*255))
                # cv2.waitKey(24)
                self.Km += self.calc_heuristic(last_node, self.curr_node)

                # after moving to next unoccupied node with lowest cost
                # update the nodes neighbouring to new walls found
                # do not update the wall nodes i.e g_val(wall_node) = inf
                nodes_to_update = []
                for obs_node in unexplored_obs:
                    for neighbour_node in self.grid_map.neighbors(obs_node):
                        if neighbour_node not in self.grid_map.obstacles:
                            nodes_to_update.append(neighbour_node)
                            
                # eliminate repetions
                nodes_to_update = set(nodes_to_update)
                self.update_nodes(nodes_to_update)

                last_node = self.curr_node
                self.compute_shortest_path()

            # for debugging
            self.render_pathplanning(
                grid_map=self.grid_map.true_map,
                explored_grid_map=self.grid_map.explored_grid_map,
                start_pt=self.start_node,
                curr_pt=self.curr_node,
                goal_pt=self.goal_node,
                path=self.get_path()
            )

            self.display_map(show_path=True)
            if self.is_new_goal:
                self.is_new_goal = False
                print("Computing new shortest path")
                print("New goal", self.true_goal_node)
                print("New grid goal", self.goal_node)
                self.compute_shortest_path()

                # for debugging
                self.render_pathplanning(
                    grid_map=self.grid_map.true_map,
                    explored_grid_map=self.grid_map.explored_grid_map,
                    start_pt=self.start_node,
                    curr_pt=self.curr_node,
                    goal_pt=self.goal_node,
                    path=self.get_path(),
                    wait_time=2000
                )
            



