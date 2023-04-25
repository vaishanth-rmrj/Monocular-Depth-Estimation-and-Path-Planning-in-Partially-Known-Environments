import numpy as np
import cv2

class MapGrid:
    def __init__(self, map_img, step_size=1, inflation_radius = 15, convert_to_grid=True):       
        
        self.walls = set()        
        if convert_to_grid:
            self.true_map = map_img
            self.true_map_with_inflation = self.add_inflation_layer(map_img, inflation_radius)
            self.true_costmap = self.compute_cost_map(self.true_map_with_inflation)
            self.true_map_vertex, self.grid_map = self.convert_to_grid(self.true_costmap/255)  

        else:
            self.grid_map = map_img
            
        self.height, self.width = self.grid_map.shape
        self.step_size = step_size   
    
    def add_inflation_layer(self, map_img, inflation_radius):

        # Calculate the distance transform
        dist_transform = cv2.distanceTransform(map_img, cv2.DIST_L2, 3)

        # Apply dilation operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_radius, inflation_radius))
        inflated_dist_transform = cv2.dilate(dist_transform, kernel)
        opening = cv2.morphologyEx(inflated_dist_transform, cv2.MORPH_OPEN, kernel, iterations=1)

        # Convert back to binary format
        inflated_bin = cv2.threshold(opening, inflation_radius, 1, cv2.THRESH_BINARY)[1]

        return inflated_bin.astype(np.uint8)

    def compute_cost_map(self, map_img, threshold_reduction=3):

        dist_transform = cv2.distanceTransform(map_img, cv2.DIST_L2, 3)
        img_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_dist_thres = np.where(img_normalized*threshold_reduction > 1, 1, img_normalized*threshold_reduction)

        return (img_dist_thres*255).astype(np.uint8)

    def convert_to_grid(self, map_img, grid_size = 3):
        grid_shape = (grid_size, grid_size) 
        stride = (grid_size, grid_size)

        output_shape = ((map_img.shape[0] - grid_shape[0]) // stride[0] + 1,
                        (map_img.shape[1] - grid_shape[1]) // stride[1] + 1)
        
        # dict to store the corresponding original pts of the resized map
        true_map_vertex = {}
        
        # stores resized map -> grid
        # where each pixel val is the mean of the corresponding local grid region in true map
        grid_map =np.zeros(output_shape)

        for row in range(output_shape[0]):
            for col in range(output_shape[1]): 
                # node_val = 1 if np.mean(map_img[row*stride[0]:row*stride[0]+grid_shape[0], 
                #                              col*stride[1]:col*stride[1]+grid_shape[1]]) > 0.5 else 0  
                
                node_val = np.mean(map_img[row*stride[0]:row*stride[0]+grid_shape[0], 
                                             col*stride[1]:col*stride[1]+grid_shape[1]])  
                
                node_vertex = (col*stride[1]+(grid_shape[1]//2), row*stride[0]+(grid_shape[0]//2) ) 

                grid_map[row, col] = node_val
                true_map_vertex[(row, col)] = node_vertex
        
        return true_map_vertex, grid_map

    def get_true_pt(self, pt):
        if not pt:
            # print("Returning None")
            return None
        
        if pt in self.true_map_vertex:
            # print("Returning true_map_vertex")
            return self.true_map_vertex[pt]
    
    def get_grid_pt(self, pt):
        # compare the point with the true map vertex and get its corresponding vertex in resized map
        min_dist = float('inf')
        resized_min_dist_pt = (0, 0)
        for key, other_pt in self.true_map_vertex.items():
            eucl_dist = np.sqrt((other_pt[0] - pt[0])**2 + (other_pt[1] - pt[1])**2 )
            if eucl_dist < min_dist:
                min_dist = eucl_dist
                resized_min_dist_pt = key

        # print("Returning resized_min_dist_pt")
        return resized_min_dist_pt

    
    def cost(self, from_node, to_node):
        if from_node in self.walls or to_node in self.walls:
            return float('inf')
        else:
            # print(self.grid_map[to_node[::-1]])
            return 1 + abs(1 - self.grid_map[to_node[::-1]])
            # return 1 


    def neighbors(self, vertex, no_move_direction=4):
        (x, y) = vertex
        if no_move_direction == 4:
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if no_move_direction == 8:
            results = [(x + 1, y), (x + 1, y - 1), (x, y - 1), (x -1, y - 1), 
                       (x - 1, y), (x - 1 , y + 1), (x, y+1), (x + 1, y + 1)]

        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        return list(results)
    
    def in_bounds(self, vertex):
        '''
        to check if vertex is in map bounds
        '''
        (x, y) = vertex
        return 0 <= x < self.width and 0 <= y < self.height

    def get_local_region(self, pos, view_dist=2):
        '''
        to fetch the local reg surrounding the curr pos

        return: node: 0 or 1
        0 - wall
        1 - freespace
        '''
        (px, py) = pos
        local_nodes = [(x, y) for x in range(px - view_dist, px + view_dist + 1)
                        for y in range(py - view_dist, py + view_dist + 1)
                        if self.in_bounds((x, y))]
                
        return {node: 0 if self.grid_map[(node[1], node[0])] == 0 else 1 for node in local_nodes}


class ExploredGrid(MapGrid):

    def get_unexplored_walls(self, local_reg):
        '''
        return: walls not explored already
        '''
        new_detected_walls = {node for node, node_type in local_reg.items() if node_type == 0}        
        return new_detected_walls - self.walls

    def update_walls(self, new_walls):

        # updating new walls to explored map
        for wall in new_walls:
            wall_x, wall_y = wall
            self.grid_map[(wall_y, wall_x)] = 0

        self.walls.update(new_walls)