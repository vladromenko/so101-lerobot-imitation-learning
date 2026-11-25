# SO-101 LeRobot Imitation Learning

Record teleoperated SO-101 demonstrations, validate and export them as LeRobot
datasets, train ACT and SmolVLA on a Jetson, track runs in MLflow, and evaluate
checkpoints on the physical follower. The browser manages the workflow. Models,
data, and robot control run locally on Homebrain.

**Start here: [English setup and operator guide](docs/OPERATOR_GUIDE.md).**
It covers the Jetson environment, USB and calibration setup, how to record
repeatable demonstrations, dataset splits, training settings, and physical
evaluation.

## What was tested

- Recorded 50 wrist-only demonstrations (18,960 frames) of picking up a blue
  cube and placing it in a white box. Demonstrations use calibrated
  leader-to-follower teleoperation and store RGB, measured joints, and the
  commands sent to the follower.
- Trained ACT and SmolVLA with a held-out session split. Both were later
  retrained on all 50 episodes with image augmentation.
- The final ACT checkpoint reached 30,000 updates. With its original weights,
  matched image preprocessing, and the restored LeRobot execution profile of
  50 actions per prediction, it physically picked up a cube and carried it
  almost to the box. One grasp was confirmed by the operator. Box placement,
  release, and repeatability remain to be verified.
- The final SmolVLA checkpoint reached 15,000 updates. This final fit has a
  training result; a successful physical grasp from it is unverified.

## Project layout

| Path | Purpose |
| --- | --- |
| [static/](static/) | SO-101 Studio browser interface |
| [src/so101_lab/](src/so101_lab/) | Teleoperation, recording, dataset validation, training, and policy rollout |
| [src/so101_lab/lerobot_plugin/](src/so101_lab/lerobot_plugin/) | LeRobot image preprocessing and rollout diagnostics |
| [config/](config/) | Environment snapshot, hardware calibration snapshots, and service definitions |
| [scripts/](scripts/) | Dataset export, service startup, and checkpoint maintenance |

Studio uses port `8020` and MLflow uses port `5020`. Live controls should stay
on the local machine or a trusted LAN/VPN. The services start with robot
movement disabled until an explicit operator action.

Raw demonstrations, checkpoints, logs, and the MLflow database are local
artifacts. A usable checkpoint transfer includes its entire directory, with
weights, configuration, and processors. See the [operator guide](docs/OPERATOR_GUIDE.md#artifacts-and-troubleshooting)
for local paths.

Wristlab is the separate geometry-based manipulation project. Its history lives
on the [Wristlab branch](https://github.com/vladromenko/so101-lerobot-imitation-learning/tree/codex/so101-wristlab).
[Historical Homebrain notes in Russian](README_RU.md) preserve earlier
implementation details and diagnostics.
