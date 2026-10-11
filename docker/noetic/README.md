# ViTFly ROS Noetic container

The original simulator targets Ubuntu 20.04 and ROS 1 Noetic. Keep ROS and
Flightmare in this container; train models with the host's CUDA-enabled
PyTorch.

From the repository root:

```bash
just noetic-image
just noetic-build
just noetic-check
just noetic-rollout 1 state environment_141
just noetic-collect
```

Catkin build, devel, and log spaces are persistent Docker volumes. The source
checkout is bind-mounted at `/ws/src/vitfly`, so expert trajectories appear in
`envtest/ros/train_set` on the host.

The environment argument selects one of the supplied tree scenes explicitly.
Multiple rollouts in one invocation reuse that scene; use distinct environment
numbers when collecting a diverse fine-tuning set.

`noetic-collect` runs the deterministic expert across a curated set of
centerline-tree scenes. Only collision-free trajectories with aligned images
and telemetry are retained under `training/datasets/tree_dead_on`.

The rollout command uses NVIDIA CDI (`nvidia.com/gpu=all`) and the current X11
authorization file so the Unity renderer can use the host GPU under XWayland.
