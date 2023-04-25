import rospy
import cv2
import torch

import numpy as np
import sys
import time
from midas.model_loader import default_models, load_model
import os
import rospkg
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import open3d


class MonocluarDepthEstimator:
    def __init__(self):
        # params
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/image_depth')
        self.point_cloud_topic = rospy.get_param('~point_cloud_topic', '/camera/point_cloud')  

        # model params
        self.is_optimize = rospy.get_param('~optimize', False)
        self.is_square = rospy.get_param('~square', False)
        self.is_grayscale = rospy.get_param('~grayscale', False)
        self.height = rospy.get_param('~height', None)
        self.model_type = rospy.get_param('~model_type', "midas_v21_small_256")
        self.model_weights_path = rospy.get_param('~model_weights_path', "/weights/midas_v21_small_256.pt")
        self.is_debug = rospy.get_param('~debug', False)
        self.frame_id = rospy.get_param("~frame_id", "base_footprint")

        # Depth camera parameters:
        self.fx_depth = 5.8262448167737955e+02
        self.fy_depth = 5.8269103270988637e+02
        self.cx_depth = 3.1304475870804731e+02
        self.cy_depth = 2.3844389626620386e+02

        # ros point cloud vars
        self.point_field_xyz = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self.point_field_xyzrgb = self.point_field_xyz + [
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # cv bridge 
        self.cv_bridge = CvBridge()

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running inference on : %s" % self.device)

        # Read pytorch model
        self.rospack = rospkg.RosPack()
        self.model_path = self.rospack.get_path("monocular_depth_estimator") + self.model_weights_path

        # loading model
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, self.model_path, 
                                                                        self.model_type, self.is_optimize, 
                                                                        self.height, self.is_square)    
        print("Net width and height: ", (self.net_w, self.net_h))

        # subscribers
        self.rgb_cam_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_cam_callback)

        # publishers
        self.depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=1)
        self.point_cloud_pub = rospy.Publisher(self.point_cloud_topic, PointCloud2, queue_size=1)

    def generate_point_cloud_msg(self, depth_img):

        depth_img *= 255

        # get depth resolution:
        print(depth_img.shape)
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
        
        # converting point cloud data to ros msg
        pcd_msg = PointCloud2()
        pcd_msg.header.stamp = rospy.Time.now()
        pcd_msg.header.frame_id = self.frame_id
        # pcd_msg.header.seq = self.counter
        pcd_msg.header.seq = 0

        # Message data size
        pcd_msg.height = 1
        pcd_msg.width = width * height

        # Fields of the point cloud
        pcd_msg.fields = [
            PointField("y", 0, PointField.FLOAT32, 1),
            PointField("z", 4, PointField.FLOAT32, 1),
            PointField("x", 8, PointField.FLOAT32, 1),
        #     PointField("b", 12, PointField.FLOAT32, 1),
        #     PointField("g", 16, PointField.FLOAT32, 1),
        #     PointField("r", 20, PointField.FLOAT32, 1)
        ]

        pcd_msg.is_bigendian = False
        pcd_msg.point_step = 24
        pcd_msg.row_step = pcd_msg.point_step * height * width
        pcd_msg.is_dense = True
        pcd_msg.data = pcd.tobytes()        
        
        return pcd_msg

    def publish_point_cloud(self, color_img, depth_img, camera_info, is_color=True, depth_unit=0.001, depth_trunc=3.0):

        # generate rgbd point cloud from color and depth image
        # type: open3d.geometry.RGBDImage
        rgbd_img = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color = open3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)),
            depth = open3d.geometry.Image(depth_img),
            depth_scale= 1.0/depth_unit,
            convert_rgb_to_intensity = False   
        )

        '''
        Intrinsic matrix format:
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        '''
        width, height = camera_info['width'], camera_info['height']
        im = np.array(camera_info['intrinsic_matrix']).reshape((3,3))
        open3d_cam_info = open3d.camera.PinholeCameraIntrinsic(
            width, height, im[0,0], im[1, 1], im[0, 2], im[1, 2]
        )

        open3d_point_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(
            image = rgbd_img,
            intrinsic = open3d_cam_info
        )

        # convert open3d point cloud to ros PointCloud2 msg
        pcd_msg = PointCloud2()
        pcd_msg.header.stamp = rospy.Time.now()
        pcd_msg.header.frame_id = self.frame_id
        # pcd_msg.header.seq = self.counter
        pcd_msg.header.seq = 0

        pcd = np.asarray(open3d_point_cloud.points)
        if not is_color:
            ros_pcd_fields = self.point_field_xyz
        else:
            ros_pcd_fields = self.point_field_xyzrgb
            # change rgb color from 'three float to 'one 24-byte int'
            colors = np.floor(np.asarray(open3d_point_cloud.colors)*255)

            BIT_MOVE_16 = 2**16
            BIT_MOVE_8 = 2**8
            colors = colors[:, 0] * BIT_MOVE_16 \
                        + colors[:, 1] * BIT_MOVE_8 \
                        + colors[:, 2]
            
            pcd = np.c_[pcd, colors]
        
        pcd_msg.fields = ros_pcd_fields
        pcd_msg.is_bigendian = False
        pcd_msg.point_step = 24
        pcd_msg.row_step = pcd_msg.point_step * height * width
        pcd_msg.is_dense = True

        # print(pcd)
        pcd_msg.data = pcd.tobytes()
        # print(pcd_msg.data)

        self.point_cloud_pub.publish(pcd_msg)


    
    def publish_depth_img(self, depth_img):

        # convert img to uint8
        cv2_uint16_depth_img = depth_img.astype(np.uint8)
        assert(type(cv2_uint16_depth_img[0, 0]) == np.uint8)

        # convert opencv image to ros img msg
        # note: do not change encoding type
        ros_depth_img = self.cv_bridge.cv2_to_imgmsg(cv2_uint16_depth_img, encoding='8UC1')

        # adding header to ros message
        ros_depth_img.header = Header()
        ros_depth_img.header.stamp = rospy.Time.now()
        ros_depth_img.header.frame_id = self.frame_id
        self.depth_pub.publish(ros_depth_img)

    def predict(self, image, model, target_size):        

        # convert img to tensor and load to gpu
        img_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)

        if self.is_optimize and self.device == torch.device("cuda"):
            img_tensor = img_tensor.to(memory_format=torch.channels_last)
            img_tensor = img_tensor.half()
        
        prediction = model.forward(img_tensor)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return prediction

    def process_prediction(self, original_img, depth_img, is_grayscale=False, side_by_side=False):
        """
        Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
        for better visibility.
        Args:
            original_img: the RGB image
            depth_img: the depth map
            is_grayscale: use a grayscale colormap?
        Returns:
            the image and depth map place side by side
        """

        # normalizing depth image
        depth_min = depth_img.min()
        depth_max = depth_img.max()
        depth_side = 255 * (depth_img - depth_min) / (depth_max - depth_min)
        
        if self.is_debug:
            depth_side_3dim = np.repeat(np.expand_dims(depth_side, 2), 3, axis=2)

            if not is_grayscale:
                depth_side_3dim = cv2.applyColorMap(np.uint8(depth_side_3dim), cv2.COLORMAP_INFERNO)

            return depth_side, np.concatenate((original_img, depth_side_3dim), axis=1)/255       
            
        return depth_side, None
           
    

    
    def rgb_cam_callback(self, ros_rgb_img_msg):
        
        # converting image to opencv image
        try:
            rgb_img_cv = self.cv_bridge.imgmsg_to_cv2(ros_rgb_img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(e)

        # if self.is_debug:
        #     cv2.imshow("RGB frame", rgb_img_cv)
        #     cv2.waitKey(1)

        # Get image data as a numpy array to be passed for processing.
        img = cv2.cvtColor(rgb_img_cv, cv2.COLOR_BGR2RGB)

        with torch.no_grad():                
            inference_start_time = time.time()              

            original_image_rgb = np.flip(img, 2)  # in [0, 255] (flip required to get RGB)
            # resizing the image to feed to the model
            image_tranformed = self.transform({"image": original_image_rgb/255})["image"]

            # monocular depth prediction
            prediction = self.predict(image_tranformed, self.model, target_size=original_image_rgb.shape[1::-1])
            # original_image_bgr = np.flip(original_image_rgb, 2) if self.side_by_side else None   
            original_image_bgr = np.flip(original_image_rgb, 2) 

            # process the model predictions
            depth_img_out, side_by_side_out = self.process_prediction(original_image_bgr, prediction, is_grayscale=self.is_grayscale)

            self.publish_depth_img(depth_img_out)

            # camera_info = {
            #     "width": 640,
            #     "height": 480,
            #     "intrinsic_matrix": [5.8262448167737955e+02, 0, 3.1304475870804731e+02, 0, 5.8269103270988637e+02, 2.3844389626620386e+02, 0, 0, 1]
            # }

            # self.publish_point_cloud(original_image_bgr, depth_img_out, camera_info, 
            #                          depth_unit=0.001, depth_trunc=3.0)

            # point_cloud = self.generate_point_cloud_msg(depth_img_out)
            # self.point_cloud_pub.publish(point_cloud)

            
            if self.is_debug:
                inference_end_time = time.time()
                fps = round(1/(inference_end_time - inference_start_time))
                cv2.putText(side_by_side_out, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 100), 2)
                cv2.imshow('MiDaS Depth Estimation', side_by_side_out)
                cv2.waitKey(24)

                    


if __name__ == '__main__':
    rospy.init_node("monocular_depth_estimator")

    depth_estimator = MonocluarDepthEstimator()
    rospy.spin()