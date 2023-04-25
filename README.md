# Monocular-Depth-Estimation-and-Path-Planning-in-Partially-Known-Environments

This is a project that utilizes monocular depth estimation to build a 3D map of partially known environments and plan paths for robots to navigate through the environment. The project is implemented using Python, PyTorch, OpenCV, NumPy, ROS, Gazebo, and RViz.

## Features
- Monocular depth estimation using deep learning
- 3D mapping using the MiDas ML algorithm
- Path planning using the D* Lite algorithm
- Simulation of robots in the Gazebo environment
- Visualization of the 3D map and robot path in RViz

## Requirements
- Python 3.8
- PyTorch
- OpenCV
- NumPy
- ROS Noetic
- Gazebo
- RViz

## Installation
1. Clone the repository

```
git clone https://github.com/vaishanth-rmrj/Monocular-Depth-Estimation-and-Path-Planning-in-Partially-Known-Environments.git
```

2. Install the required dependencies
```
pip install torch opencv-python numpy
```

3. Install ROS and Gazebo. Follow the instructions on the official ROS and Gazebo websites for installation.
4. Build the ROS packages in the `catkin_ws` directory

```
cd ros_mono_depth_ws
catkin_make
```

5. Launch the node using
```
roslaunch robot_launcher simulation_system.launch
```

## Credits
- [MiDas: A PyTorch implementation of the paper "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"](https://github.com/intel-isl/MiDaS)
- [D* Lite path planning algorithm implementation in Python](https://github.com/PedroHenrique-git/D-Star-Lite)
- [Gazebo simulation environment](http://gazebosim.org/)
- [RViz visualization tool](http://wiki.ros.org/rviz)

