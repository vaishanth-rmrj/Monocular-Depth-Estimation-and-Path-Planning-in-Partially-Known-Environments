from grid import ExploredGrid, MapGrid
from priority_queue import PriorityQueue
from collections import deque
from functools import partial
from utility import convert_grid_to_img
import cv2
import numpy as np

class DStarLite(object):
    def __init__(self, map_grid, start, goal, view_range=2):
        self.start_node = start
        self.curr_node = start
        self.goal_node = goal

        # Init the graphs
        self.explored_grid = ExploredGrid(grid_string="", width=map_grid.width, height=map_grid.height)
        self.org_mapgrid = map_grid
        self.view_range = view_range

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

    def get_rhsval(self, node):
        return self.rhs_val.get(node, float('inf'))
    
    def calc_rhs_val(self, node):
        lowest_cost_neighbour = self.fetch_lowest_cost_neighbour(node)
        return self.calc_lookahead_cost(node, lowest_cost_neighbour), lowest_cost_neighbour

    def fetch_lowest_cost_neighbour(self, node):
        neighbours = self.explored_grid.neighbors(node)
        cost_fn = partial(self.calc_lookahead_cost, node)
        return min(neighbours, key=cost_fn)

    def calc_lookahead_cost(self, node, neighbour):
        return self.get_gval(neighbour) + self.explored_grid.cost(node, neighbour)

    def update_nodes(self, nodes):
        for node in nodes:
            if node != self.goal_node:
                # setting the rhs value of the node
                # and setting the backpointer of the node
                # to the lowest cost neighbour
                self.rhs_val[node], self.back_pointers[node] = self.calc_rhs_val(node)
            
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
        
        last_nodes = deque(maxlen=10)
        # loop until 2 conditions are satisfied
        # 1. current / start node is locally consistent
        # 2. there exists no node with key val less than curr node
        while self.queue.top_key() < self.calc_key(self.curr_node) or self.g_val.get(self.curr_node) != self.rhs_val.get(self.curr_node):
            
            k_old = self.queue.top_key()
            node = self.queue.pop_smallest()[1]

            # fail safe
            last_nodes.append(node)
            if len(last_nodes) == 10 and len(set(last_nodes)) < 3:
                print("Stuck in loop")
                raise Exception("Stuck in a loop")

            k_new = self.calc_key(node_pt=node)
            

            if k_new > k_old:
                # nodes key value has changed because of obstacle
                self.queue.insert(node, k_new)
            elif self.get_gval(node) > self.get_rhsval(node):
                # since lookahead val of node is less than its gval
                # we can reach the node with the min val of rhs
                self.g_val[node] = self.get_rhsval(node)

                # updating neighbour node
                neighbours = self.explored_grid.neighbors(node)
                self.update_nodes(neighbours)
            else:
                self.g_val[node] = float('inf')
                neighbours = self.explored_grid.neighbors(node)
                self.update_nodes(neighbours+[node])

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
    
    def render_pathplanning(self, map_grid, explored_map_grid, start_pt, curr_pt, goal_pt, path):
        map_grid = map_grid.copy()
        explored_map_grid = explored_map_grid.copy()

        # processing true map grid
        # adding start and goal points
        map_grid[(start_pt[1], start_pt[0])]= 0.5
        map_grid[(goal_pt[1], goal_pt[0])]= 0.6
        map_img = convert_grid_to_img(map_grid)

        # processing explored map grid
        # adding path points
        for point in path:
            explored_map_grid[(point[1], point[0])] = 0.7
        # adding start and goal points
        explored_map_grid[(curr_pt[1], curr_pt[0])]= 0.5
        explored_map_grid[(goal_pt[1], goal_pt[0])]= 0.6
        explored_map_img = convert_grid_to_img(explored_map_grid)

        # side by side img
        border_line = np.zeros((map_img.shape[0], 5, 3), dtype=np.uint8)
        side_by_side_img = np.concatenate((map_img, border_line), axis=1)
        side_by_side_img = np.concatenate((side_by_side_img, explored_map_img), axis=1)
        cv2.imshow("Map Grid - Explored Map Grid: (Press q for next)", side_by_side_img)
        if cv2.waitKey(200) == ord('q'):
            return 0
            
    def move_and_replan(self):

        # get local region for the curr node
        local_reg = self.org_mapgrid.get_local_region(self.curr_node, self.view_range)
        unexplored_walls = self.explored_grid.get_unexplored_walls(local_reg)
        self.explored_grid.update_walls(unexplored_walls)

        # compute shortest path to goal traverses from goal node to start node
        # this path is with the explored obstacles
        self.compute_shortest_path()
        
        # for debugging
        self.render_pathplanning(
            map_grid=self.org_mapgrid.map_img,
            explored_map_grid=self.explored_grid.map_img,
            start_pt=self.start_node,
            curr_pt=self.curr_node,
            goal_pt=self.goal_node,
            path=[]
        )
        
        last_node = self.curr_node
        while self.curr_node != self.goal_node:                      

            if self.get_gval(self.curr_node) == float('inf'):
                raise Exception("no path found")

            self.curr_node = self.fetch_lowest_cost_neighbour(self.curr_node)
            local_roi = self.org_mapgrid.get_local_region(self.curr_node, self.view_range)
            new_walls = self.explored_grid.get_unexplored_walls(local_roi)

            if new_walls:
                self.explored_grid.update_walls(new_walls)
                self.Km += self.calc_heuristic(last_node, self.curr_node)

                # after moving to next unoccupied node with lowest cost
                # update the nodes neighbouring to new walls found
                # do not update the wall nodes i.e g_val(wall_node) = inf
                nodes_to_update = []
                for wall_node in new_walls:
                    for neighbour_node in self.explored_grid.neighbors(wall_node):
                        if neighbour_node not in self.explored_grid.walls:
                            nodes_to_update.append(neighbour_node)
                # eliminate repetions
                nodes_to_update = set(nodes_to_update)
                self.update_nodes(nodes_to_update)

                last_node = self.curr_node
                self.compute_shortest_path()

            # for debugging
            self.render_pathplanning(
                map_grid=self.org_mapgrid.map_img,
                explored_map_grid=self.explored_grid.map_img,
                start_pt=self.start_node,
                curr_pt=self.curr_node,
                goal_pt=self.goal_node,
                path=self.get_path()
            )
            



