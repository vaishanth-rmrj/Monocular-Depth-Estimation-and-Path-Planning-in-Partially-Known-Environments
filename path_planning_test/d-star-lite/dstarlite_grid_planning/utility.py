import numpy as np

def convert_grid_to_img(grid, start_val = 0.5, goal_val = 0.6, path_val= 0.7):
    '''
    Convert numpy grid to resized grid image

    Values map:
    Start point: 0.5
    Goal point: 0.6
    Path points: 0.7
    '''
    grid_img = grid.copy()    
    grid_img = (grid_img*255).astype(np.uint8)    

    # converting to 3channel image
    rgb_img = np.zeros((grid_img.shape[0], grid_img.shape[1], 3), dtype=np.uint8)
    rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2] = grid_img, grid_img, grid_img

    # seting diff color vals for start, end and path point    
    start_pt_val = int(start_val*255)
    goal_pt_val = int(goal_val*255)
    path_pt_val = int(path_val*255)
    rgb_img = np.where(rgb_img == [path_pt_val, path_pt_val, path_pt_val], [251, 140, 0], rgb_img)
    rgb_img = np.where(rgb_img == [goal_pt_val, goal_pt_val, goal_pt_val], [60, 142, 56], rgb_img)
    rgb_img = np.where(rgb_img == [start_pt_val, start_pt_val, start_pt_val], [255, 109, 83], rgb_img)
    grid_img = rgb_img.astype(np.uint8)

    # expanding/resizing the grid for better visibility
    grid_img = np.repeat(grid_img, 20, axis=0)
    grid_img = np.repeat(grid_img, 20, axis=1)

    return grid_img

