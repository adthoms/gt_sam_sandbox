# gt_sam_sandbox

A repository for experiments conducted with GTSAM

---

## Dependency

See the following [link](https://github.com/BEAMRobotics/beam_robotics/wiki/Beam-Robotics-Installation-Guide) for installing the following dependencies:

- install_cmake
- install_qwt
- install_catch2
- install_eigen3
- install_ceres

Additionally, [basalt-headers-mirror](git@github.com:BEAMRobotics/basalt-headers-mirror.git) is required

---

## Install

Use the following commands to download and compile the package.

```shell
cd ~/catkin_ws/src
git clone git@github.com:BEAMRobotics/basalt-headers-mirror.git
git clone git@github.com:adthoms/gt_sam_sandbox.git
cd ..
catkin build
```

---
