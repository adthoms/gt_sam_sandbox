# gt_sam_sandbox

A repository for experiments conducted with GTSAM

---

## Dependency

See the following [link](https://github.com/BEAMRobotics/beam_robotics/wiki/Beam-Robotics-Installation-Guide) for installing the following dependencies:

- install_cmake
- install_eigen3
- install_gtsam

---

## Install

Use the following commands to download and compile the package.

```shell
cd ~/catkin_ws/src
git clone git@github.com:adthoms/gt_sam_sandbox.git
cd ..
catkin build
```

---

## Run

To run executables:

```shell
cd ~/catkin_ws/devel/lib/gt_sam_sandbox
./gt_sam_sandbox_iot_slam_concept_main
```

---

## TODOs

- generate factors between landmarks (representing iot devices)
- calibrate between factor noise to represent noise expected of LIO/VIO/LVIO odometry system
