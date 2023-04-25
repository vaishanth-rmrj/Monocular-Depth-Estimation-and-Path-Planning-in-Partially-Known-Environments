import numpy as np

def render_to_rgb(normalized_map):
    '''
    Convert numpy grid to resized grid image
    '''
    map_rgb_img = normalized_map.copy()    
    map_rgb_img = (map_rgb_img*255).astype(np.uint8)    

    # converting to 3channel image
    rgb_img = np.zeros((map_rgb_img.shape[0], map_rgb_img.shape[1], 3), dtype=np.uint8)
    rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2] = map_rgb_img, map_rgb_img, map_rgb_img
    map_rgb_img = rgb_img.astype(np.uint8)

    return map_rgb_img

