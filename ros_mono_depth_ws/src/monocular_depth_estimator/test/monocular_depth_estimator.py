import cv2
import torch
import numpy as np
import time
from midas.model_loader import default_models, load_model
import os

class MonocularDepthEstimator:
    def __init__(self, input_path, 
                    model_type="midas_v21_small_256",
                    model_weights_path="weights/midas_v21_small_256.pt", 
                    optimize=False, 
                    side_by_side=True, 
                    height=None, 
                    square=False, 
                    grayscale=True):

        # model type
        # MiDaS 3.1:
        # For highest quality: dpt_beit_large_512
        # For moderately less quality, but better speed-performance trade-off: dpt_swin2_large_384
        # For embedded devices: dpt_swin2_tiny_256, dpt_levit_224
        # For inference on Intel CPUs, OpenVINO may be used for the small legacy model: openvino_midas_v21_small .xml, .bin
        
        # MiDaS 3.0: 
        # Legacy transformer models dpt_large_384 and dpt_hybrid_384

        # MiDaS 2.1: 
        # Legacy convolutional models midas_v21_384 and midas_v21_small_256
        
        # params
        print("Initializing parameters and model...")
        self.is_optimize = optimize
        self.is_square = square
        self.is_grayscale = grayscale
        self.height = height
        self.side_by_side = side_by_side
        self.input_path = "input/testvideo2.mp4"

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running inference on : %s" % self.device)

        # loading model
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_weights_path, 
                                                                        model_type, optimize, height, square)    
        print("Net width and height: ", (self.net_w, self.net_h))
        

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
        normalized_depth = 255 * (depth_img - depth_min) / (depth_max - depth_min)
        normalized_depth *= 3

        depth_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
        if not is_grayscale:
            depth_side = cv2.applyColorMap(np.uint8(depth_side), cv2.COLORMAP_INFERNO)

        if side_by_side:
            return np.concatenate((original_img, depth_side), axis=1)/255       
            
        return depth_side/255

    def run(self):
        
        # input video
        cap = cv2.VideoCapture(self.input_path)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video file")

        with torch.no_grad():
             while cap.isOpened():

                # Capture frame-by-frame
                inference_start_time = time.time()
                ret, frame = cap.read()                

                if ret == True:
                    original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
                    # resizing the image to feed to the model
                    image_tranformed = self.transform({"image": original_image_rgb/255})["image"]

                    # monocular depth prediction
                    prediction = self.predict(image_tranformed, self.model, target_size=original_image_rgb.shape[1::-1])
                    original_image_bgr = np.flip(original_image_rgb, 2) if self.side_by_side else None   

                    # process the model predictions
                    output = self.process_prediction(original_image_bgr, prediction, is_grayscale=self.is_grayscale, side_by_side=self.side_by_side)
                    
                    inference_end_time = time.time()
                    fps = round(1/(inference_end_time - inference_start_time))
                    cv2.putText(output, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 100), 2)
                    cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', output)

                    # Press ESC on keyboard to exit
                    if cv2.waitKey(1) == 27:  # Escape key
                        break
                
                else:
                    break


        # When everything done, release
        # the video capture object
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()



if __name__ == "__main__":
    # params
    INPUT_PATH = "input/testvideo2.mp4"

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

     # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    depth_estimator = MonocularDepthEstimator(INPUT_PATH, side_by_side=False)
    depth_estimator.run()

