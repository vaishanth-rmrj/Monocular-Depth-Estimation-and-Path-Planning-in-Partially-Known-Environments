import numpy as np
import cv2

class MapGrid:
    def __init__(self, 
                true_map, 
                grid_block_size = 3, 
                stride_size = 3,
                inflation_radius = 0):     

        self.true_map = true_map
        self.true_vertex_mapping = {}
        self.inflation_radius = inflation_radius

        # initializing the expored grid map
        # the code will resize the true map to much smaller size for faster computation
        self.grid_shape = (grid_block_size, grid_block_size) 
        self.stride_shape = (stride_size, stride_size)
        # mean pool output shape for the resized grid map
        grid_output_shape = ((true_map.shape[0] - self.grid_shape[0]) // self.stride_shape[0] + 1,
                        (true_map.shape[1] - self.grid_shape[1]) // self.stride_shape[1] + 1)
        
        self.explored_grid_map = np.ones(grid_output_shape, dtype=np.uint8)
        self.unaltered_explored_grid_map = np.ones(grid_output_shape, dtype=np.uint8)
        self.explored_grid_costmap = np.ones(grid_output_shape, dtype=np.uint8)
        self.height, self.width = grid_output_shape
        
        self.obstacles = set()         
    
    def add_inflation_layer(self, map_img, inflation_radius):

        # Calculate the distance transform
        dist_transform = cv2.distanceTransform(map_img, cv2.DIST_L2, 3)

        # Apply dilation operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_radius, inflation_radius))
        inflated_dist_transform = cv2.dilate(dist_transform, kernel)
        # opening = cv2.morphologyEx(inflated_dist_transform, cv2.MORPH_OPEN, kernel, iterations=1)
        # print(inflated_dist_transform)

        # Convert back to binary format
        inflated_bin = cv2.threshold(inflated_dist_transform, inflation_radius, 1, cv2.THRESH_BINARY)[1]
        # print(inflated_bin)

        return inflated_bin.astype(np.uint8)

    def compute_cost_map(self, map_img, threshold_reduction=3):

        if np.argwhere(map_img==0).shape[0] > 100:

            dist_transform = cv2.distanceTransform(map_img, cv2.DIST_L2, 3)
            img_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_dist_thres = np.where(img_normalized*threshold_reduction > 1, 1, img_normalized*threshold_reduction)

            return img_dist_thres

        return map_img
    
    def update_grid_inflation_and_costmap(self):
        # print("Updating grid cost map")
        # self.explored_grid_costmap = self.compute_cost_map(self.explored_grid_map.astype(np.uint8), threshold_reduction=3)
        # print("Updated grid cost map")

        # adding inflation layer
        self.explored_grid_map = self.add_inflation_layer(self.explored_grid_map.astype(np.uint8), 20)

    

    def get_true_pt(self, pt):        
        if not pt:
            return None        
        if pt in self.true_vertex_mapping:
            return self.true_vertex_mapping[pt]
        else:
            # calc true map point based on grid block shape
            pt_x = (pt[0] + (self.grid_shape[0] // 2)) * self.stride_shape[0]
            pt_y = (pt[1] + (self.grid_shape[1] // 2)) * self.stride_shape[1]
            return (pt_x, pt_y)
        
    def get_grid_pt(self, pt):
        # compare the point with the true map vertex and get its corresponding vertex in resized map
        # Scale down the x and y coordinates of the robot pose by a factor of k
        scaled_x = pt[0] / self.grid_shape[0]
        scaled_y = pt[1] / self.grid_shape[1]

        # Subtract grid_block_size/2 from the scaled x and y coordinates to account for the mean pooling offset
        grid_x = int(scaled_x - (self.grid_shape[0] / 2))
        grid_y = int(scaled_y - (self.grid_shape[1] / 2))

        return (grid_x, grid_y)

    
    def cost(self, from_node, to_node):
        if from_node in self.obstacles or to_node in self.obstacles:
            return float('inf')
        else:
            # print(abs(1 - self.explored_grid_costmap[to_node[::-1]]))
            return 1 + abs(1 - self.explored_grid_costmap[to_node[::-1]])
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
    
    def convert_to_grid(self, pixel_arr, true_pixel_arr_index, grid_pose, grid_shape = (3, 3), stride_shape = (3, 3)):

        output_shape = ((pixel_arr.shape[0] - grid_shape[0]) // stride_shape[0] + 1,
                        (pixel_arr.shape[1] - grid_shape[1]) // stride_shape[1] + 1)
                
        # stores resized map -> grid
        # where each pixel val is the mean of the corresponding local grid region in true map
        output_pixel_arr =np.zeros(output_shape)
        gx, gy = grid_pose
        output_grid_arr_index = np.dstack(np.meshgrid(np.arange(gy-output_shape[0]//2, (gy+output_shape[0]//2)+1), 
                                                      np.arange(gx-output_shape[1]//2, (gx+output_shape[1]//2)+1), indexing='ij'))

        for row in range(output_shape[0]):
            for col in range(output_shape[1]): 
                node_val = 1 if np.mean(pixel_arr[row*stride_shape[0]:row*stride_shape[0]+grid_shape[0], 
                                             col*stride_shape[1]:col*stride_shape[1]+grid_shape[1]]) > 0.5 else 0  
                
                # node_val = np.mean(map_img[row*stride_shape[0]:row*stride_shape[0]+grid_shape[0], 
                #                              col*stride_shape[1]:col*stride_shape[1]+grid_shape[1]])  
                
                node_vertex = (col*stride_shape[1]+(grid_shape[1]//2), row*stride_shape[0]+(grid_shape[0]//2) ) 

                output_pixel_arr[row, col] = node_val
                # print("row, col",row, col)
                # print(output_grid_arr_index[(row, col)])
                # print(tuple(output_grid_arr_index[(row, col)]))
                # print(true_pixel_arr_index.shape)
                # print(true_pixel_arr_index[node_vertex])
                self.true_vertex_mapping[tuple(output_grid_arr_index[(row, col)])[::-1]] = tuple(true_pixel_arr_index[(row, col)])
        
        return output_pixel_arr
    
    def in_bounds(self, vertex):
        '''
        to check if vertex is in map bounds
        '''
        (x, y) = vertex
        return 0 <= x < self.width and 0 <= y < self.height

    def get_local_region(self, true_pos, view_dist):

        # pos wrt to true map
        px, py = true_pos
        # print(true_pos)
        # pos wrt to grid map
        # Scale down the x and y coordinates of the robot pose by a factor of k
        # Subtract grid_block_size/2 from the scaled x and y coordinates to account for the mean pooling offset
        scaled_x, scaled_y = px / self.grid_shape[0], py / self.grid_shape[1]
        gx, gy = int(scaled_x - (self.grid_shape[0] / 2)), int(scaled_y - (self.grid_shape[1] / 2))

        # getting local region pxs on true map
        w, h = 2*view_dist, 2*view_dist

        # calculate the top-left corner of the rectangle
        x_tl = int(px - (w / 2))
        y_tl = int(py - (h / 2))

        # check if the pixels' spatial positions are within the image bounds
        # create a meshgrid of the pixel positions
        yy, xx = np.meshgrid(np.arange(y_tl, (y_tl+h)+1), np.arange(x_tl, (x_tl+w)+1), indexing='ij')
        # create a mask for pixels within the image bounds        
        mask = (yy >= 0) & (yy < self.true_map.shape[0]) & (xx >= 0) & (xx < self.true_map.shape[1])        
        # apply the mask to the pixel positions to get the positions within the image bounds
        yy, xx = yy[mask], xx[mask]
        
        # set the limits to slice from true map
        left_limit, right_limit = min(xx), max(xx)
        upper_limit, lower_limit = min(yy), max(yy)

        # use the masked pixel positions to access the corresponding pixel values in the region
        local_region_pixels = self.true_map[upper_limit:lower_limit, left_limit:right_limit]
        
        # cv2.imshow("local_region_pixels_inflated", (local_region_pixels*255).astype(np.uint8))
        # cv2.waitKey(24)

        true_map_pixel_index = np.dstack(np.meshgrid(np.arange(upper_limit, lower_limit), np.arange(left_limit, right_limit ), indexing='ij'))
        

        grid_pixels = self.convert_to_grid(local_region_pixels, 
                                           true_map_pixel_index, 
                                           grid_pose=(gx, gy),
                                           grid_shape=self.grid_shape, 
                                           stride_shape=self.stride_shape)
        
        grid_arr_index = np.dstack(np.meshgrid(np.arange(gy-grid_pixels.shape[0]//2, (gy+(grid_pixels.shape[0]//2))+1), 
                                               np.arange(gx-grid_pixels.shape[1]//2, (gx+(grid_pixels.shape[1]//2))+1), indexing='ij'))
        
        return local_region_pixels, grid_pixels, true_map_pixel_index, grid_arr_index, (gx, gy)


    def update_explored_grid(self, grid_local_region, true_map_vertex, grid_arr_vertex, curr_pose, grid_block_size=3):

        grid_x, grid_y = curr_pose
        height, width = grid_local_region.shape

        # old 
        # self.true_map_with_inflation = self.add_inflation_layer(map_img, inflation_radius)
        # self.true_costmap = self.compute_cost_map(self.true_map_with_inflation)
        # self.true_map_vertex, self.grid_map = self.convert_to_grid(self.true_costmap/255) 

        # adding inflation layer to local region
        # cv2.imshow("Grid ", grid_local_region)
        # grid_local_region = self.add_inflation_layer(grid_local_region.astype(np.uint8), self.inflation_radius)
        # grid_local_region = self.compute_cost_map(grid_local_region.astype(np.uint8), threshold_reduction=1)
        # print(grid_local_region)
        # cv2.imshow("Grid with inflation", grid_local_region*255)
        # cv2.waitKey(0)


        # updating the grid map
        self.explored_grid_map[grid_y-height//2: (grid_y+height//2)+1, grid_x-width//2: (grid_x+width//2)+1] = grid_local_region
        # print(self.compute_cost_map(grid_local_region.astype(np.uint8), threshold_reduction=3))
        # self.explored_grid_costmap = self.compute_cost_map(self.explored_grid_map.astype(np.uint8), threshold_reduction=3)
        # self.unaltered_explored_grid_map = self.explored_grid_map
        

        # updating true vertex mapping
        row_ids, col_ids = np.meshgrid(np.arange(grid_arr_vertex.shape[0]), 
                                        np.arange(grid_arr_vertex.shape[1]), indexing='ij')
        row_ids = row_ids*self.stride_shape[0]+(self.grid_shape[0]//2)
        col_ids = col_ids*self.stride_shape[1]+(self.grid_shape[1]//2)
        true_vertex = true_map_vertex[row_ids, col_ids]

        true_vertex_flipped = np.flip(true_vertex, axis=2)
        grid_arr_vertex_flipped = np.flip(grid_arr_vertex, axis=2)

        true_vertex_reordered = true_vertex_flipped.flatten().reshape(-1, 2)
        grid_arr_vertex_reordered = grid_arr_vertex_flipped.flatten().reshape(-1, 2)

        keys = tuple(map(tuple, grid_arr_vertex_reordered))
        values = tuple(map(tuple, true_vertex_reordered))
        self.true_vertex_mapping.update(dict(zip(keys, values)))
        
        # getting obstacle points with respect to explored grid
        obs_pts = np.argwhere(grid_local_region == 0)
        obs_pts = grid_arr_vertex[tuple(obs_pts.T)]
        # print(obs_pts)
        # reversing the pt from row, col (y, x) to x, y
        obs_pts = set(map(tuple, np.flip(obs_pts, axis=1)))

        # print(obs_pts)


        # testing
        # true_map = self.true_map.copy()
        # for row in range(true_vertex_mapping.shape[0]):
        #     for col in range(true_vertex_mapping.shape[1]):
        #         y, x = int(true_vertex_mapping[(row, col)][0]), int(true_vertex_mapping[(row, col)][1])
        #         # print(y, x)
        #         true_map[(y, x)] = 0
        
        # grid_map = self.explored_grid_map.copy()
        # for row in range(grid_arr_vertex.shape[0]):
        #     for col in range(grid_arr_vertex.shape[1]):
        #         y, x = int(grid_arr_vertex[(row, col)][0]), int(grid_arr_vertex[(row, col)][1])
        #         # print(y, x)
        #         grid_map[(y, x)] = 0

        
        # for obs_pt in obs_pts:
        #     grid_map[obs_pt[::-1]] = 0


        # # cv2.imshow("true map", true_map*255)
        # cv2.imshow("grid map", grid_map*255)
        # cv2.imshow("exp grid map", self.explored_grid_map*255)

        # cv2.waitKey(0)
        # assert(1 > 2)

        return obs_pts

        

    
    def get_unexplored_obstacles(self, obs_pt):
        '''
        return: obstacles not explored already
        '''
        obs_tuple_arr = tuple(map(tuple, obs_pt))
        obs_set_arr = set(obs_tuple_arr)
            
        return obs_set_arr - self.obstacles

    def update_obstacles(self, new_obstacles):     
        self.obstacles.update(new_obstacles) 


class ExploredGrid(MapGrid):

    # def get_unexplored_obstacles_old(self, local_reg):
    #     '''
    #     return: obstacles not explored already
    #     '''
    #     new_detected_obstacles = {node for node, node_type in local_reg.items() if node_type == 0}        
    #     return new_detected_obstacles - self.obstacles
    
    def get_unexplored_obstacles(self, local_reg):
        '''
        return: obstacles not explored already
        '''
        print(local_reg)
        print(np.where(local_reg == 1))
        new_detected_obstacles = set(np.where(local_reg == 1) )     
        return new_detected_obstacles - self.obstacles

    def update_obstacles(self, new_obstacles):

        # updating new obstacles to explored map
        for wall in new_obstacles:
            wall_x, wall_y = wall
            self.grid_map[(wall_y, wall_x)] = 0

        self.obstacles.update(new_obstacles)