import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d





# print(pcd)
# skip = 100   # Skip every n points

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# point_range = range(0, pcd.shape[0], skip) # skip points to prevent crash
# ax.scatter(pcd[point_range, 0],   # x
#            pcd[point_range, 1],   # y
#            pcd[point_range, 2],   # z
#            c=pcd[point_range, 2], # height data for color
#            cmap='spectral',
#            marker="x")
# ax.axis('scaled')  # {equal, scaled}
# plt.show()

# pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
# pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
# # Visualize:
# o3d.visualization.draw_geometries([pcd_o3d])


class PointCloudGenerator:
    def __init__(self):
        # Depth camera parameters:
        self.fx_depth = 5.8262448167737955e+02
        self.fy_depth = 5.8269103270988637e+02
        self.cx_depth = 3.1304475870804731e+02
        self.cy_depth = 2.3844389626620386e+02

    def conver_to_point_cloud_v1(self, depth_img):

        pcd = []
        height, width = depth_img.shape
        for i in range(height):
            for j in range(width):
                z = depth_img[i][j]
                x = (j - self.cx_depth) * z / self.fx_depth
                y = (i - self.cy_depth) * z / self.fy_depth
                pcd.append([x, y, z])
        
        return pcd

    def conver_to_point_cloud_v2(self, depth_img):

        # get depth resolution:
        height, width = depth_img.shape
        length = height * width

        # compute indices:
        jj = np.tile(range(width), height)
        ii = np.repeat(range(height), width)

        # rechape depth image
        z = depth_img.reshape(length)
        # compute pcd:
        pcd = np.dstack([(ii - self.cx_depth) * z / self.fx_depth,
                        (jj - self.cy_depth) * z / self.fy_depth,
                        z]).reshape((length, 3))
        
        return pcd

    def generate_point_cloud(self, image_path, vectorize=False):
        depth_img = cv2.imread(image_path, 0)       

        print(f"Image resolution: {depth_img.shape}")
        print(f"Data type: {depth_img.dtype}")
        print(f"Min value: {np.min(depth_img)}")
        print(f"Max value: {np.max(depth_img)}")


        # normalizing depth image
        depth_min = depth_img.min()
        depth_max = depth_img.max()
        normalized_depth = 255 * ((depth_img - depth_min) / (depth_max - depth_min))

        depth_img = normalized_depth
        print("After normalization: ")
        print(f"Image resolution: {depth_img.shape}")
        print(f"Data type: {depth_img.dtype}")
        print(f"Min value: {np.min(depth_img)}")
        print(f"Max value: {np.max(depth_img)}")


        # convert depth to point cloud
        if not vectorize:
            self.pcd = self.conver_to_point_cloud_v1(depth_img)
        if vectorize:
            self.pcd = self.conver_to_point_cloud_v2(depth_img)

        
        return self.pcd
    
    def viz_point_cloud(self, use_matplotlib=False):
        
        points = np.array(self.pcd)
        skip = 200 
        point_range = range(0, points.shape[0], skip) # skip points to prevent crash

        if use_matplotlib:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[point_range, 0], points[point_range, 1], points[point_range, 2], c='r', marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
        
        if not use_matplotlib:

            pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
            # Visualize:
            o3d.visualization.draw_geometries([pcd_o3d])


if __name__ == "__main__":
    input_image = "test/inputs/depth.png"
    point_cloud_gen = PointCloudGenerator()
    pcd = point_cloud_gen.generate_point_cloud(input_image)
    point_cloud_gen.viz_point_cloud()



